"""
Coffee Shop Order Queue - FastAPI backend

Bridges:
  - Customer/barista HTTP requests <-> Redis (strings, hashes, lists, sorted sets)
  - Redis Pub/Sub "order_updates" channel <-> WebSocket clients (barista screen)

Run:
  uvicorn app:app --reload
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import date

import numpy as np
import redis.asyncio as redis
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from redis import Redis as SyncRedis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.schema import IndexSchema
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)

REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = int(os.environ["REDIS_PORT"])
REDIS_PASSWORD = os.environ["REDIS_PASSWORD"]
# This database has TLS disabled in the Redis Cloud console (Security > TLS: Off).
# Set REDIS_TLS=true in .env if you later enable TLS on the database.
REDIS_TLS = os.environ.get("REDIS_TLS", "false").lower() == "true"

ORDER_CHANNEL = "order_updates"
# Ready-for-pickup window (seconds). Default 600 (10 min) for real use;
# drop to something short like 20-30 for a live demo of TTL expiry.
PICKUP_TTL_SECONDS = int(os.environ.get("PICKUP_TTL_SECONDS", "600"))

# --- Redis connection (async client, for the coffee-shop endpoints) ---
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    ssl=REDIS_TLS,
    protocol=2,  # avoid redis-py 6.x RESP3/HELLO handshake quirk with Redis Cloud auth
    decode_responses=True,
)

# --- FAQ vector search setup (reuses the faq_idx index built by setup_faq_index.py) ---
# RedisVL's SearchIndex uses the sync redis client, so this is separate from the
# async client above. Same connection details, same shared database.
FAQ_EMBEDDING_DIMS = 384
FAQ_DISTANCE_THRESHOLD = float(os.environ.get("FAQ_DISTANCE_THRESHOLD", "0.6"))

faq_schema = IndexSchema.from_dict({
    "index": {
        "name": "faq_idx",
        "prefix": "faq",
        "storage_type": "hash",
    },
    "fields": [
        {"name": "id", "type": "tag"},
        {"name": "question", "type": "text"},
        {"name": "answer", "type": "text"},
        {
            "name": "question_embedding",
            "type": "vector",
            "attrs": {
                "dims": FAQ_EMBEDDING_DIMS,
                "distance_metric": "cosine",
                "algorithm": "flat",
                "datatype": "float32",
            },
        },
    ],
})

faq_sync_redis = SyncRedis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    ssl=REDIS_TLS,
    protocol=2,
    decode_responses=False,  # vector bytes must stay raw
)
faq_index = SearchIndex(faq_schema, redis_client=faq_sync_redis)

# The embedding model is loaded once at startup (see lifespan below), not per-request -
# model loading is the expensive part, encoding individual questions after that is fast.
faq_model: SentenceTransformer | None = None

# --- Track connected barista WebSocket clients ---
active_connections: list[WebSocket] = []


async def pubsub_listener():
    """Background task: subscribe to order_updates, forward to all connected WebSockets."""
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(ORDER_CHANNEL)
    async for message in pubsub.listen():
        if message["type"] != "message":
            continue
        dead = []
        for ws in active_connections:
            try:
                await ws.send_text(message["data"])
            except Exception:
                dead.append(ws)
        for ws in dead:
            active_connections.remove(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global faq_model
    print("Loading FAQ embedding model (all-MiniLM-L6-v2)...")
    faq_model = await asyncio.to_thread(SentenceTransformer, "all-MiniLM-L6-v2")
    print("FAQ embedding model loaded.")

    task = asyncio.create_task(pubsub_listener())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


# --- Request models ---
class OrderRequest(BaseModel):
    customer: str
    drink: str
    size: str
    milk: str = "none"


class FaqRequest(BaseModel):
    question: str


# --- Endpoints ---

@app.get("/health", response_class=PlainTextResponse)
async def health():
    pong = await redis_client.ping()
    return "pong" if pong else "no response"


@app.post("/order")
async def create_order(order: OrderRequest):
    counter_key = f"orders:counter:{date.today().isoformat()}"
    order_id = await redis_client.incr(counter_key)
    order_key = f"order:{order_id}"

    await redis_client.hset(
        order_key,
        mapping={
            "customer": order.customer,
            "drink": order.drink,
            "size": order.size,
            "milk": order.milk,
            "status": "queued",
            "created_at": date.today().isoformat(),
        },
    )
    await redis_client.rpush("queue:orders", order_id)
    await redis_client.zincrby("leaderboard:drinks", 1, order.drink)

    # Customer profile hash - a separate pattern from the order record hash
    # above: a small "account" object keyed by customer, tracking their
    # favorite (most recent) drink and running order count. Demo simplification:
    # slugified name as the key rather than a real customer/account ID.
    customer_key = f"customer:{order.customer.strip().lower().replace(' ', '_')}"
    await redis_client.hset(
        customer_key,
        mapping={"name": order.customer, "favorite_drink": order.drink},
    )
    total_orders = await redis_client.hincrby(customer_key, "total_orders", 1)

    await redis_client.publish(
        ORDER_CHANNEL, json.dumps({"event": "new_order", "id": order_id})
    )

    return {"order_id": order_id, "status": "queued", "customer_total_orders": total_orders}


@app.get("/customer/{name}")
async def get_customer(name: str):
    customer_key = f"customer:{name.strip().lower().replace(' ', '_')}"
    profile = await redis_client.hgetall(customer_key)
    if not profile:
        return {"found": False}
    return {"found": True, **profile}


@app.get("/queue")
async def get_queue():
    order_ids = await redis_client.lrange("queue:orders", 0, -1)
    orders = []
    for oid in order_ids:
        data = await redis_client.hgetall(f"order:{oid}")
        if data:
            data["id"] = oid
            orders.append(data)
    return orders


@app.post("/order/{order_id}/complete")
async def complete_order(order_id: str):
    await redis_client.hset(f"order:{order_id}", "status", "done")
    await redis_client.lrem("queue:orders", 1, order_id)

    # Transient "ready for pickup" window - a string with a TTL. Once this
    # key expires, the order has passed its pickup window. Separate from the
    # permanent status field in the order hash.
    await redis_client.set(f"order:{order_id}:pickup_status", "ready_for_pickup", ex=PICKUP_TTL_SECONDS)

    await redis_client.publish(
        ORDER_CHANNEL, json.dumps({"event": "order_completed", "id": order_id})
    )

    return {"order_id": order_id, "status": "done"}


@app.get("/leaderboard")
async def get_leaderboard():
    top = await redis_client.zrevrange("leaderboard:drinks", 0, 4, withscores=True)
    return [{"drink": drink, "count": int(count)} for drink, count in top]


@app.post("/faq/ask")
async def faq_ask(req: FaqRequest):
    """
    Embeds the question, runs a KNN vector search against faq_idx (built by
    setup_faq_index.py), and applies the same distance-threshold confidence
    check as query_faq.py. Runs on a background thread since sentence-transformers
    and the RedisVL sync client are both blocking calls.
    """
    def _search():
        vec = faq_model.encode(req.question).astype(np.float32).tobytes()
        vq = VectorQuery(
            vector=vec,
            vector_field_name="question_embedding",
            return_fields=["question", "answer"],
            num_results=1,
        )
        return faq_index.query(vq)

    results = await asyncio.to_thread(_search)

    if not results:
        return {"confident": False, "matched_question": None, "answer": None, "distance": None}

    top = results[0]
    distance = float(top.get("vector_distance", 1.0))
    confident = distance < FAQ_DISTANCE_THRESHOLD

    return {
        "confident": confident,
        "matched_question": top["question"],
        "answer": top["answer"] if confident else None,
        "distance": distance,
    }


@app.get("/order/{order_id}/pickup_status")
async def get_pickup_status(order_id: str):
    status = await redis_client.get(f"order:{order_id}:pickup_status")
    if status is None:
        return {"status": "expired_or_not_found", "seconds_remaining": 0}
    ttl = await redis_client.ttl(f"order:{order_id}:pickup_status")
    return {"status": status, "seconds_remaining": ttl}


@app.get("/pickup")
async def get_pickup_orders():
    """
    All orders currently in their 'ready for pickup' TTL window, for the
    barista screen's pickup panel. Uses SCAN (non-blocking, safe on a live
    server) rather than KEYS (blocks Redis while it scans the whole keyspace).
    """
    orders = []
    async for key in redis_client.scan_iter(match="order:*:pickup_status"):
        order_id = key.split(":")[1]
        status = await redis_client.get(key)
        ttl = await redis_client.ttl(key)
        if status is None or ttl is None or ttl < 0:
            continue
        details = await redis_client.hgetall(f"order:{order_id}")
        orders.append({
            "id": order_id,
            "status": status,
            "seconds_remaining": ttl,
            "customer": details.get("customer", ""),
            "drink": details.get("drink", ""),
        })
    orders.sort(key=lambda o: o["seconds_remaining"])
    return orders


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # We don't expect messages from the client; just keep the connection alive.
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)


# Serve the static frontend files (customer.html, barista.html) from ./static
# Guarded so the backend still runs standalone (e.g. testing /health, /docs)
# before the static/ folder and frontend files exist.
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

