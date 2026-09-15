"""
FAQ Vector Store - setup script

Creates the RediSearch vector index (via RedisVL) on the shared Redis Cloud
database, embeds a small coffee-shop FAQ dataset with sentence-transformers,
loads it, and runs one self-verification query.

Run:
  python setup_faq_index.py
"""

import os

import numpy as np
from dotenv import load_dotenv
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.schema import IndexSchema
from sentence_transformers import SentenceTransformer

load_dotenv()

REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = int(os.environ["REDIS_PORT"])
REDIS_PASSWORD = os.environ["REDIS_PASSWORD"]
REDIS_TLS = os.environ.get("REDIS_TLS", "false").lower() == "true"

EMBEDDING_DIMS = 384  # matches all-MiniLM-L6-v2

# --- FAQ dataset (coffee shop themed) ---
FAQ_ENTRIES = [
    {"id": "1", "question": "What time do you open?",
     "answer": "We're open Monday-Friday 6am-7pm, and weekends 7am-6pm."},
    {"id": "2", "question": "Do you have oat milk?",
     "answer": "Yes, we offer oat, almond, soy, and coconut milk at no extra charge."},
    {"id": "3", "question": "Are your pastries gluten-free?",
     "answer": "We have a few gluten-free pastry options, but our kitchen isn't a dedicated gluten-free facility."},
    {"id": "4", "question": "How do I earn rewards?",
     "answer": "You earn 1 point per dollar spent through our app; 100 points gets you a free drink."},
    {"id": "5", "question": "Can I order ahead?",
     "answer": "Yes, you can place an order through our app and pick it up in-store without waiting in line."},
    {"id": "6", "question": "Do you deliver?",
     "answer": "We partner with DoorDash and Uber Eats for delivery within a 5-mile radius."},
    {"id": "7", "question": "Is there parking nearby?",
     "answer": "There's a public lot directly behind the shop, and metered street parking out front."},
    {"id": "8", "question": "Do you take Apple Pay?",
     "answer": "Yes, we accept Apple Pay, Google Pay, and all major credit cards."},
    {"id": "9", "question": "Do you have wifi?",
     "answer": "Free wifi is available for customers; the password is posted at the counter."},
    {"id": "10", "question": "Can I bring my dog?",
     "answer": "Well-behaved dogs are welcome on our outdoor patio, but not inside."},
]

# --- RedisVL schema: index faq_idx, key prefix faq: (keeps this data namespaced
#     alongside the coffee-shop keys on the same shared free-tier database) ---
schema = IndexSchema.from_dict({
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
                "dims": EMBEDDING_DIMS,
                "distance_metric": "cosine",
                "algorithm": "flat",
                "datatype": "float32",
            },
        },
    ],
})


def main():
    redis_client = Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        ssl=REDIS_TLS,
        protocol=2,  # avoid redis-py 6.x RESP3/HELLO handshake quirk with Redis Cloud auth
        decode_responses=False,  # vector bytes must stay raw
    )
    print("PING:", redis_client.ping())

    index = SearchIndex(schema, redis_client=redis_client)
    # overwrite=True makes this script safely re-runnable during demo prep
    index.create(overwrite=True, drop=True)
    print(f"Created index '{schema.index.name}' with prefix '{schema.index.prefix}'")

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Embedding and loading {len(FAQ_ENTRIES)} FAQ entries...")
    records = []
    for entry in FAQ_ENTRIES:
        vec = model.encode(entry["question"]).astype(np.float32)
        records.append({
            "id": entry["id"],
            "question": entry["question"],
            "answer": entry["answer"],
            "question_embedding": vec.tobytes(),
        })
    index.load(records, id_field="id")
    print(f"Loaded {len(records)} entries into '{schema.index.name}'.")

    # --- Self-verification query ---
    test_question = "Is the shop open on Sundays?"
    print(f"\nSelf-verification query: \"{test_question}\"")
    query_vec = model.encode(test_question).astype(np.float32).tobytes()

    vq = VectorQuery(
        vector=query_vec,
        vector_field_name="question_embedding",
        return_fields=["question", "answer"],
        num_results=1,
    )
    results = index.query(vq)
    if results:
        top = results[0]
        print(f"  Top match: \"{top['question']}\"")
        print(f"  Answer: {top['answer']}")
        print(f"  Distance: {top.get('vector_distance', 'n/a')}  (lower = more similar, cosine)")
    else:
        print("  No results returned - check index/data load.")

    info = index.info()
    print(f"\nIndex info: {info.get('num_docs', '?')} docs indexed in '{schema.index.name}'.")


if __name__ == "__main__":
    main()
