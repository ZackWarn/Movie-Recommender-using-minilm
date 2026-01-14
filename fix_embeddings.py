"""
Fix the embeddings issue by regenerating them using the local BERT model.
This script will generate proper embeddings instead of all-zero vectors.
"""

import os
import sys

# Temporarily disable external embeddings to force local model usage
os.environ["HF_SPACE_ENDPOINT"] = ""
os.environ["USE_EXTERNAL_EMBEDDINGS"] = "false"

from data_prep import load_and_prepare_data
from bert_processor import MovieBERTProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_embeddings():
    """Regenerate embeddings using local BERT model"""
    logger.info("=" * 60)
    logger.info("FIXING EMBEDDINGS - Regenerating with Local BERT Model")
    logger.info("=" * 60)

    try:
        # Load movie data
        logger.info("\n1. Loading movie data...")
        movies_data = load_and_prepare_data()
        logger.info(f"   ✓ Loaded {len(movies_data)} movies")

        # Initialize BERT processor with local model
        logger.info("\n2. Initializing BERT processor (local model)...")
        processor = MovieBERTProcessor(lazy_load=False)

        # Force enable local model by directly loading it
        logger.info("\n3. Loading local BERT model...")
        from sentence_transformers import SentenceTransformer

        processor._model = SentenceTransformer(processor.model_name)
        logger.info(f"   ✓ Loaded model: {processor.model_name}")

        # Generate embeddings
        logger.info("\n4. Generating embeddings...")
        logger.info("   This may take several minutes...")
        embeddings = processor.generate_embeddings(movies_data)
        logger.info(f"   ✓ Generated embeddings with shape: {embeddings.shape}")

        # Verify embeddings are not all zeros
        import numpy as np

        if np.all(embeddings == 0):
            logger.error("   ❌ ERROR: Generated embeddings are all zeros!")
            return False

        # Check embedding statistics
        logger.info(f"\n   Embedding statistics:")
        logger.info(f"   - Mean: {embeddings.mean():.6f}")
        logger.info(f"   - Std: {embeddings.std():.6f}")
        logger.info(f"   - Min: {embeddings.min():.6f}")
        logger.info(f"   - Max: {embeddings.max():.6f}")

        # Save embeddings
        logger.info("\n5. Saving embeddings to file...")
        processor.save_embeddings("movie_embeddings.pkl")
        logger.info("   ✓ Embeddings saved successfully")

        # Verify by loading and testing
        logger.info("\n6. Verifying saved embeddings...")
        processor2 = MovieBERTProcessor(lazy_load=True)
        processor2.load_embeddings()
        test_embeddings = processor2._get_embeddings()

        if np.all(test_embeddings == 0):
            logger.error("   ❌ ERROR: Saved embeddings are all zeros!")
            return False

        logger.info(f"   ✓ Verification successful")
        logger.info(f"   ✓ Loaded embeddings shape: {test_embeddings.shape}")

        logger.info("\n" + "=" * 60)
        logger.info("✅ EMBEDDINGS FIXED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("\nYou can now test the recommendation system:")
        logger.info("  python flask_api.py")
        logger.info("\n")

        return True

    except Exception as e:
        logger.error(f"\n❌ Error fixing embeddings: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = fix_embeddings()
    sys.exit(0 if success else 1)
