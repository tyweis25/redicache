"""
FAQ Vector Store - query script

Runs the paraphrase test queries against the faq_idx index built by
setup_faq_index.py: a keyword-search pass (to show its limits) followed by
a vector KNN pass (to show semantic matching), plus one out-of-scope query
to demonstrate a similarity-threshold confidence check.

Assumes setup_faq_index.py has already been run.

Run:
  python query_faq.py
"""

import os

import numpy as np
from dotenv import load_dotenv
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.schema import IndexSchema
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)

REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = int(os.environ["REDIS_PORT"])
REDIS_PASSWORD = os.environ["REDIS_PASSWORD"]
REDIS_TLS = os.environ.get("REDIS_TLS", "false").lower() == "true"

EMBEDDING_DIMS = 384

# Placeholder threshold - tuned provisionally against the scores we've seen
# so far (real match ~0.4, expect an out-of-scope query to score meaningfully
# higher/worse). Cosine DISTANCE here: lower = more similar.
DISTANCE_THRESHOLD = float(os.environ.get("FAQ_DISTANCE_THRESHOLD", "0.6"))

# Same schema as setup_faq_index.py - must match exactly to query the same index
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

TEST_QUERIES = [
    "Is the shop open on Sundays?",
    "I'm lactose intolerant, what are my options?",
    "Can I sit outside with my pet?",
    "Can I return a used mug?",  # deliberately out-of-scope
]


def keyword_search(redis_client, query_text):
    """Plain TEXT search on the question field, no vector involved."""
    # RediSearch's query parser treats punctuation like ? and ' as syntax,
    # so strip it for this simple demo query (a production system would use
    # proper query escaping instead).
    safe_text = "".join(c for c in query_text if c.isalnum() or c.isspace())
    try:
        result = redis_client.execute_command(
            "FT.SEARCH", "faq_idx", f'@question:"{safe_text}"', "LIMIT", "0", "3"
        )
        # result[0] is the total number of matches
        return result[0]
    except Exception as e:
        return f"error: {e}"


def main():
    redis_client = Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        ssl=REDIS_TLS,
        protocol=2,
        decode_responses=False,
    )

    index = SearchIndex(schema, redis_client=redis_client)

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    for query_text in TEST_QUERIES:
        print(f"\n{'=' * 70}")
        print(f"QUERY: \"{query_text}\"")
        print("=" * 70)

        # --- Keyword search pass ---
        kw_match_count = keyword_search(redis_client, query_text)
        print(f"  Keyword search matches: {kw_match_count}")

        # --- Vector search pass ---
        query_vec = model.encode(query_text).astype(np.float32).tobytes()
        vq = VectorQuery(
            vector=query_vec,
            vector_field_name="question_embedding",
            return_fields=["question", "answer"],
            num_results=3,
        )
        results = index.query(vq)

        if not results:
            print("  Vector search: no results returned.")
            continue

        top = results[0]
        distance = float(top.get("vector_distance", 1.0))

        if distance < DISTANCE_THRESHOLD:
            print(f"  Vector search top match: \"{top['question']}\" (distance: {distance:.3f})")
            print(f"    -> Answer: {top['answer']}")
        else:
            print(f"  Vector search: NO CONFIDENT MATCH (top distance: {distance:.3f}, "
                  f"threshold: {DISTANCE_THRESHOLD})")
            print(f"    Closest topic was: \"{top['question']}\" - treating as low-confidence.")

        print("\n  Top 3 candidates:")
        for r in results:
            print(f"    - \"{r['question']}\" (distance: {float(r.get('vector_distance', -1)):.3f})")


if __name__ == "__main__":
    main()
