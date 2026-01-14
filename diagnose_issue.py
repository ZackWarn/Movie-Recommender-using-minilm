"""Diagnose the recommendation issue"""

import numpy as np
import pickle
import os

print("=" * 60)
print("DIAGNOSTIC CHECK FOR RECOMMENDATION ISSUE")
print("=" * 60)

# Check if embeddings file exists
embeddings_file = "movie_embeddings.pkl"
if not os.path.exists(embeddings_file):
    print(f"\n❌ ERROR: {embeddings_file} not found!")
    exit(1)

print(f"\n✓ Found embeddings file: {embeddings_file}")

# Load embeddings
with open(embeddings_file, "rb") as f:
    data = pickle.load(f)

embeddings = data.get("embeddings")
movies_data = data.get("movies_data")
pca = data.get("pca")

print(f"\n📊 Embeddings shape: {embeddings.shape}")
print(f"📊 Number of movies: {len(movies_data)}")
print(f"📊 PCA components: {pca.n_components_ if pca else 'None'}")

# Check if all embeddings are the same
print("\n🔍 Checking embedding diversity...")
unique_embeddings = np.unique(embeddings, axis=0)
print(f"   Unique embedding vectors: {len(unique_embeddings)}")

if len(unique_embeddings) == 1:
    print("   ❌ PROBLEM: All embeddings are identical!")
elif len(unique_embeddings) < 10:
    print(
        f"   ⚠️  WARNING: Only {len(unique_embeddings)} unique embeddings for {len(embeddings)} movies"
    )
else:
    print(f"   ✓ Embeddings are diverse ({len(unique_embeddings)} unique)")

# Check embedding statistics
print("\n📈 Embedding statistics:")
print(f"   Mean: {embeddings.mean():.6f}")
print(f"   Std: {embeddings.std():.6f}")
print(f"   Min: {embeddings.min():.6f}")
print(f"   Max: {embeddings.max():.6f}")

# Check for all-zero embeddings
all_zeros = np.all(embeddings == 0, axis=1)
if all_zeros.any():
    print(f"   ❌ WARNING: {all_zeros.sum()} movies have all-zero embeddings!")

# Sample a few movies and their embeddings
print("\n🎬 Sample movies:")
for i in range(min(5, len(movies_data))):
    movie = movies_data.iloc[i]
    emb_norm = np.linalg.norm(embeddings[i])
    print(f"   {i}: {movie['clean_title']} - embedding norm: {emb_norm:.4f}")

# Test similarity computation
print("\n🧪 Testing similarity computation...")
from sklearn.metrics.pairwise import cosine_similarity

# Create a test query embedding (random)
test_query = np.random.randn(1, embeddings.shape[1]).astype(np.float32)
similarities = cosine_similarity(test_query, embeddings)[0]

print(f"   Similarity range: [{similarities.min():.4f}, {similarities.max():.4f}]")
print(f"   Similarity mean: {similarities.mean():.4f}")
print(f"   Similarity std: {similarities.std():.4f}")

# Get top 8
top_indices = similarities.argsort()[::-1][:8]
print(f"\n   Top 8 indices: {top_indices}")
print(f"   Top 8 scores: {similarities[top_indices]}")

# Check if top 8 are always the same
print("\n🔄 Testing with different random queries...")
all_top_indices = []
for test_num in range(5):
    test_query = np.random.randn(1, embeddings.shape[1]).astype(np.float32)
    similarities = cosine_similarity(test_query, embeddings)[0]
    top_indices = similarities.argsort()[::-1][:8]
    all_top_indices.append(set(top_indices))

# Check if all tests return the same movies
if len(set.intersection(*all_top_indices)) == 8:
    print("   ❌ PROBLEM: Same 8 movies recommended for ALL queries!")
    print(f"   Always recommended: {list(set.intersection(*all_top_indices))}")
    # Show which movies these are
    common_indices = list(set.intersection(*all_top_indices))
    print("\n   Movies always recommended:")
    for idx in common_indices:
        movie = movies_data.iloc[idx]
        print(f"      {idx}: {movie['clean_title']} ({movie.get('year', 'Unknown')})")
else:
    print(f"   ✓ Different queries return different results")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
