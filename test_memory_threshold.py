"""Test memory thresholds to find optimal configuration for Render deployment"""

import logging
import sys
import gc
import psutil
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_memory():
    """Get current process memory in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_memory_usage():
    """Test memory usage through different stages"""
    print("\n" + "=" * 70)
    print("MEMORY USAGE TEST - Simulating Render Free Tier (512MB limit)")
    print("=" * 70)

    RENDER_LIMIT_MB = 512

    # Stage 1: Startup
    print(f"\n1. STARTUP (bare Python)")
    startup_mb = get_memory()
    print(f"   Memory: {startup_mb:.2f} MB")
    print(f"   Headroom: {RENDER_LIMIT_MB - startup_mb:.2f} MB")
    print(f"   Status: {'✅ SAFE' if startup_mb < RENDER_LIMIT_MB else '❌ EXCEEDED'}")

    # Stage 2: Import modules
    print(f"\n2. IMPORTING MODULES")
    from bert_processor import MovieBERTProcessor
    from rec_engine import MovieRecommendationEngine
    from data_prep import load_and_prepare_data

    import_mb = get_memory()
    print(f"   Memory: {import_mb:.2f} MB (+{import_mb - startup_mb:.2f} MB)")
    print(f"   Headroom: {RENDER_LIMIT_MB - import_mb:.2f} MB")
    print(f"   Status: {'✅ SAFE' if import_mb < RENDER_LIMIT_MB else '❌ EXCEEDED'}")

    # Stage 3: Initialize processor (lazy load)
    print(f"\n3. INITIALIZING BERT PROCESSOR (lazy load)")
    processor = MovieBERTProcessor(lazy_load=True)
    processor_mb = get_memory()
    print(f"   Memory: {processor_mb:.2f} MB (+{processor_mb - import_mb:.2f} MB)")
    print(f"   Headroom: {RENDER_LIMIT_MB - processor_mb:.2f} MB")
    print(
        f"   Status: {'✅ SAFE' if processor_mb < RENDER_LIMIT_MB else '❌ EXCEEDED'}"
    )

    # Stage 4: Load metadata
    print(f"\n4. LOADING MOVIE METADATA")
    processor.movies_data = load_and_prepare_data()
    metadata_mb = get_memory()
    print(f"   Memory: {metadata_mb:.2f} MB (+{metadata_mb - processor_mb:.2f} MB)")
    print(f"   Headroom: {RENDER_LIMIT_MB - metadata_mb:.2f} MB")
    print(f"   Status: {'✅ SAFE' if metadata_mb < RENDER_LIMIT_MB else '❌ EXCEEDED'}")
    print(f"   Movies loaded: {len(processor.movies_data):,}")

    # Stage 5: Initialize engine
    print(f"\n5. INITIALIZING RECOMMENDATION ENGINE")
    engine = MovieRecommendationEngine(processor, use_imdb=False)
    engine_mb = get_memory()
    print(f"   Memory: {engine_mb:.2f} MB (+{engine_mb - metadata_mb:.2f} MB)")
    print(f"   Headroom: {RENDER_LIMIT_MB - engine_mb:.2f} MB")
    print(f"   Status: {'✅ SAFE' if engine_mb < RENDER_LIMIT_MB else '❌ EXCEEDED'}")

    # Stage 6: Test keyword query
    print(f"\n6. TESTING KEYWORD QUERY (no BERT model)")
    test_query = "action thriller"
    print(f"   Query: '{test_query}'")
    results = engine.recommend_by_query(test_query, top_k=8)
    keyword_mb = get_memory()
    print(f"   Memory: {keyword_mb:.2f} MB (+{keyword_mb - engine_mb:.2f} MB)")
    print(f"   Headroom: {RENDER_LIMIT_MB - keyword_mb:.2f} MB")
    print(f"   Status: {'✅ SAFE' if keyword_mb < RENDER_LIMIT_MB else '❌ EXCEEDED'}")
    print(f"   Results returned: {len(results)}")
    if results:
        print(f"   First result: {results[0]['title']}")
        print(
            f"   Semantic used: {'✅ YES' if results[0].get('similarity_score', 0) > 0 else '❌ NO (keyword only)'}"
        )

    # Stage 7: Test with different thresholds
    print(f"\n7. TESTING DIFFERENT SEMANTIC THRESHOLDS")
    print(f"   Current memory: {keyword_mb:.2f} MB")
    print(f"   BERT model size: ~150 MB")

    for threshold in [470, 480, 490, 500, 510]:
        projected = keyword_mb + 150
        would_fit = projected <= threshold
        safe = projected < RENDER_LIMIT_MB

        print(f"\n   Threshold: {threshold} MB")
        print(f"   Projected: {projected:.2f} MB")
        print(f"   Would load BERT: {'✅ YES' if would_fit else '❌ NO'}")
        print(f"   Would exceed Render limit: {'❌ YES' if not safe else '✅ NO'}")

        if would_fit and safe:
            print(f"   ⭐ RECOMMENDED: {threshold} MB threshold")
            recommended_threshold = threshold
            break

    # Stage 8: Force garbage collection and retry
    print(f"\n8. TESTING WITH GARBAGE COLLECTION")
    gc.collect()
    gc_mb = get_memory()
    print(f"   Memory after GC: {gc_mb:.2f} MB (freed {keyword_mb - gc_mb:.2f} MB)")
    print(f"   Projected with BERT: {gc_mb + 150:.2f} MB")
    print(
        f"   Would be safe: {'✅ YES' if (gc_mb + 150) < RENDER_LIMIT_MB else '❌ NO'}"
    )

    # Final summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Render Limit:              {RENDER_LIMIT_MB} MB")
    print(f"Peak memory (keyword):     {keyword_mb:.2f} MB")
    print(f"After GC:                  {gc_mb:.2f} MB")
    print(f"Projected with BERT:       {gc_mb + 150:.2f} MB")
    print(f"Safety margin:             {RENDER_LIMIT_MB - (gc_mb + 150):.2f} MB")

    if (gc_mb + 150) < RENDER_LIMIT_MB:
        print(f"\n✅ SEMANTIC SEARCH FEASIBLE")
        optimal = min(490, int(gc_mb + 150 + 10))  # Add 10MB buffer
        print(f"   Recommended threshold: {optimal} MB")
    else:
        print(f"\n❌ SEMANTIC SEARCH NOT FEASIBLE ON FREE TIER")
        print(f"   Recommend: Keyword-only mode")

    print("=" * 70)

    return {
        "startup": startup_mb,
        "metadata": metadata_mb,
        "engine": engine_mb,
        "keyword_query": keyword_mb,
        "after_gc": gc_mb,
        "projected_semantic": gc_mb + 150,
        "results_count": len(results),
        "semantic_used": (
            results[0].get("similarity_score", 0) > 0 if results else False
        ),
    }


if __name__ == "__main__":
    try:
        results = test_memory_usage()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
