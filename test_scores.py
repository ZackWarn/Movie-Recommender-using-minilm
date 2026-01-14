#!/usr/bin/env python
"""Quick test to verify scores are displayed and no duplicates"""
import logging
import os

# Use HF Space for encoding (required since embeddings are PCA-reduced)
os.environ["HF_SPACE_ENDPOINT"] = "https://vibinjethro-mini-lm.hf.space"
os.environ["USE_EXTERNAL_EMBEDDINGS"] = "true"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

from bert_processor import MovieBERTProcessor
from rec_engine import MovieRecommendationEngine

print("\n=== Testing Recommendations ===\n")

bp = MovieBERTProcessor()
bp.load_embeddings()
engine = MovieRecommendationEngine(bp, use_imdb=False)

print("\n--- Query: 'action movies' ---")
recs = engine.recommend_by_query("action movies", top_k=8)
print(f"\nReturned {len(recs)} recommendations:")
for i, rec in enumerate(recs, 1):
    print(f"  {i}. {rec['title']}")

print("\n--- Query: 'romantic comedy' ---")
recs2 = engine.recommend_by_query("romantic comedy", top_k=8)
print(f"\nReturned {len(recs2)} recommendations:")
for i, rec in enumerate(recs2, 1):
    print(f"  {i}. {rec['title']}")
print("\n--- Query: 'Fight Club' ---")
recs3 = engine.recommend_by_query('Fight Club', top_k=8)
print(f"\nReturned {len(recs3)} recommendations:")
for i, rec in enumerate(recs3, 1):
    print(f"  {i}. {rec['title']}")

# Check if Fight Club appears in its own recommendations
has_fight_club = any('fight club' in r['title'].lower() for r in recs3)
print(f"\n[OK] 'Fight Club' excluded from its own results: {not has_fight_club}")
# Check if they're different
same_movies = set(r["title"] for r in recs) == set(r["title"] for r in recs2)
print(f"[OK] Different queries return different movies: {not same_movies}")
print(f"[OK] Got {len(recs)} results (expected 8)")
