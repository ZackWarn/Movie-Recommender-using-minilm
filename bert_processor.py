# bert_processor.py
"""BERT processor for generating and loading movie embeddings (local-only)."""

import logging
import os
import pickle
from typing import List
import psutil
import gc

from data_prep import normalize_title

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from config import Config

logger = logging.getLogger(__name__)


class MovieBERTProcessor:
    def __init__(self, model_name: str = None, lazy_load: bool = False):
        # Always use external embeddings; do not load local model
        self.model_name = model_name or Config.BERT_MODEL_NAME
        self._model = None
        self.movie_embeddings = None
        self.movies_data = None
        self.use_external = True
        self.pca = None  # PCA transformer for 32D query encoding

    def _get_memory_mb(self):
        """Get current process memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0

    def _can_safely_load_model(self, max_total_mb=400):
        """
        Check if we can safely load BERT model without exceeding limits.
        With TinyBERT (~60MB), threshold 400MB keeps us below 512MB cap.

        Args:
            max_total_mb: Maximum total memory allowed (default 450MB for safety)

        Returns:
            True if current + model_overhead <= max_total_mb
        """
        # If keyword-only mode is enabled, never load BERT
        if Config.KEYWORD_ONLY_MODE:
            logger.info("KEYWORD_ONLY_MODE enabled - skipping BERT model loading")
            return False

        current_mb = self._get_memory_mb()
        # TinyBERT overhead: 2MB if already loaded, ~60MB if loading fresh
        model_overhead = 2 if self._model is not None else 60
        projected_mb = current_mb + model_overhead
        safe = projected_mb <= max_total_mb

        logger.info(
            f"Memory check: {current_mb:.1f}MB current, "
            f"{projected_mb:.1f}MB projected (overhead: {model_overhead}MB), "
            f"{'SAFE' if safe else 'UNSAFE'} (limit: {max_total_mb}MB)"
        )

        return safe

    @property
    def model(self) -> SentenceTransformer:
        """Get or load the local BERT model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading local BERT model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str], force_semantic=False):
        """
        Encode texts using external HF Space embeddings or local model as fallback.
        """
        if not isinstance(texts, list):
            texts = [texts]

        # Try external embeddings first if configured
        if Config.HF_SPACE_ENDPOINT:
            try:
                logger.info(
                    "External encode start",
                    extra={
                        "endpoint": Config.HF_SPACE_ENDPOINT,
                        "count": len(texts),
                    },
                )
                return self._encode_external(texts)
            except Exception as e:
                logger.warning(
                    f"External encoding failed: {e}, falling back to local model"
                )

        # Fallback to local model
        logger.info("Using local BERT model for encoding")
        return self._encode_local(texts)

    def _encode_external(self, texts: List[str]):
        """
        Encode texts using external API.
        Supports both:
        1. HF Inference API (Config.HF_INFERENCE_ENDPOINT)
        2. Custom HF Space endpoint (Config.HF_SPACE_ENDPOINT)
        """
        import requests
        import time

        # Try HF Space endpoint first if configured
        endpoint = (
            getattr(Config, "HF_SPACE_ENDPOINT", None) or Config.HF_INFERENCE_ENDPOINT
        )
        headers = {}
        if Config.HF_API_TOKEN:
            headers["Authorization"] = f"Bearer {Config.HF_API_TOKEN}"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # For HF Space, use /embed endpoint
                if hasattr(Config, "HF_SPACE_ENDPOINT") and Config.HF_SPACE_ENDPOINT:
                    url = f"{endpoint.rstrip('/')}/embed"
                    payload = {"texts": texts}
                    response = requests.post(
                        url, json=payload, headers=headers, timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()
                        embeddings = result.get("embeddings", [])
                        embeddings_array = np.array(embeddings, dtype=np.float32)

                        # If using PCA-reduced embeddings, transform to same dimensionality
                        if self.pca is not None:
                            embeddings_reduced = self.pca.transform(embeddings_array)
                            logger.info(
                                f"HF Space API success: encoded {len(embeddings)} texts, "
                                f"reduced to {embeddings_reduced.shape[1]}D using PCA"
                            )
                            return embeddings_reduced

                        logger.info(
                            f"HF Space API success: encoded {len(embeddings)} texts"
                        )
                        return embeddings_array
                    else:
                        logger.warning(
                            f"HF Space error {response.status_code}, retrying..."
                        )
                        time.sleep(2**attempt)
                        continue
                else:
                    # Standard HF Inference API
                    response = requests.post(
                        endpoint,
                        headers=headers,
                        json={"inputs": texts, "options": {"wait_for_model": True}},
                        timeout=30,
                    )
                    if response.status_code == 200:
                        embeddings = response.json()
                        logger.info("HF Inference API success")
                        return np.array(embeddings)
                    elif response.status_code == 503:
                        logger.info("HF API 503 (model loading), retrying")
                        time.sleep(2**attempt)
                        continue
                    else:
                        logger.warning(f"HF API error {response.status_code}")
                        break
            except Exception as e:
                logger.warning(f"External API error: {e}, retrying...")
                time.sleep(2**attempt)

        # Fallback: return zeros to trigger keyword-only matching (no local model)
        raise RuntimeError("External embeddings failed after retries")

    def _encode_local(self, texts: List[str]):
        """
        Encode texts using local BERT model.
        """
        # Load model if not already loaded
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading local BERT model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)

        # Encode texts
        embeddings = self._model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # Apply PCA if available
        if self.pca is not None:
            embeddings = self.pca.transform(embeddings)
            logger.info(
                f"Local encoding complete: {len(texts)} texts reduced to {embeddings.shape[1]}D"
            )
        else:
            logger.info(f"Local encoding complete: {len(texts)} texts")

        return embeddings

    def prepare_movie_texts(self, movies_df):
        """Combine movie information into text descriptions"""
        movie_texts = []

        for _, movie in movies_df.iterrows():
            text_parts = [
                f"Title: {movie['clean_title']}",
                f"Genres: {', '.join(movie['genres_list']) if movie['genres_list'] != ['(no genres listed)'] else 'Unknown'}",
                f"Year: {movie['year'] if pd.notna(movie['year']) else 'Unknown'}",
                f"Rating: {movie['avg_rating']:.1f}/5.0 ({movie['rating_count']} reviews)",
            ]

            if movie["combined_tags"]:
                tags = [str(tag) for tag in movie["combined_tags"] if pd.notna(tag)]
                text_parts.append(f"Tags: {', '.join(tags[:10])}")

            movie_texts.append(". ".join(text_parts))

        return movie_texts

    def generate_embeddings(self, movies_df):
        """Generate BERT embeddings for all movies"""
        print("Preparing movie texts...")
        movie_texts = self.prepare_movie_texts(movies_df)

        print(f"Generating embeddings for {len(movie_texts)} movies...")
        batch_size = getattr(Config, "ENCODING_BATCH_SIZE", 32) or 32
        embeddings = []

        for i in range(0, len(movie_texts), batch_size):
            batch = movie_texts[i : i + batch_size]
            # Force semantic encoding during generation so we persist real vectors
            batch_embeddings = self.encode(batch, force_semantic=True)
            embeddings.append(batch_embeddings)

            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + len(batch)}/{len(movie_texts)} movies...")

        self.movie_embeddings = np.vstack(embeddings)

        # Apply PCA dimensionality reduction: 384D -> 32D (saves ~86% memory)
        print(
            f"Reducing embeddings from {self.movie_embeddings.shape[1]}D to 32D using PCA..."
        )
        self.pca = PCA(n_components=32)
        self.movie_embeddings = self.pca.fit_transform(self.movie_embeddings).astype(
            np.float32
        )
        print(f"Embeddings reduced to {self.movie_embeddings.shape}")

        self.movies_data = movies_df.reset_index(drop=True)

        print("Embeddings generated successfully!")
        return self.movie_embeddings

    def save_embeddings(self, filepath="movie_embeddings.pkl"):
        """Save embeddings, movie data, and PCA transformer"""
        data = {
            "embeddings": self.movie_embeddings,
            "movies_data": self.movies_data,
            "pca": self.pca,  # Save PCA transformer for query encoding
        }
        resolved_path = (
            filepath
            if os.path.isabs(filepath)
            else os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)
        )
        with open(resolved_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Embeddings saved to {resolved_path}")

    def load_embeddings(self, filepath="movie_embeddings.pkl"):
        """Load pre-computed embeddings with sparse on-demand loading"""
        # Skip if already loaded
        if self.movies_data is not None:
            return

        candidate_path = (
            filepath
            if os.path.isabs(filepath)
            else os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)
        )
        if not os.path.exists(candidate_path):
            alt_path = os.path.abspath(filepath)
            if os.path.exists(alt_path):
                candidate_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Embeddings file not found at '{filepath}' or '{candidate_path}'"
                )

        with open(candidate_path, "rb") as f:
            data = pickle.load(f)

        embeddings = data["embeddings"]

        # Store embeddings filepath and metadata only, defer actual embedding array loading
        self._embeddings_file = candidate_path
        self._embeddings_shape = embeddings.shape
        self._embeddings_dtype = embeddings.dtype

        # Load PCA transformer if present (for query encoding)
        self.pca = data.get("pca", None)
        if self.pca is not None:
            logger.info(f"Loaded PCA transformer: {self.pca.n_components_}D reduction")

        # Store only the movie data, not embeddings
        self.movie_embeddings = None
        self.movies_data = data["movies_data"].copy()

        # Normalize titles (e.g., "Dark Knight, The" -> "The Dark Knight")
        if "clean_title" in self.movies_data.columns:
            self.movies_data["clean_title"] = self.movies_data["clean_title"].apply(
                normalize_title
            )
        if "title" in self.movies_data.columns:
            self.movies_data["title"] = self.movies_data["title"].apply(normalize_title)

        # Downcast numeric columns to save metadata memory
        for col in self.movies_data.columns:
            col_type = self.movies_data[col].dtype
            if col_type == "float64":
                self.movies_data[col] = self.movies_data[col].astype("float32")
            elif col_type == "int64":
                self.movies_data[col] = self.movies_data[col].astype("int32")

        print(
            f"Embeddings metadata loaded from {candidate_path} (embeddings loaded on-demand)"
        )

    def _get_embeddings(self):
        """Lazy load embeddings on-demand"""
        if self.movie_embeddings is None:
            print("Loading embeddings from disk...")
            self._log_memory("before loading embeddings from disk")

            with open(self._embeddings_file, "rb") as f:
                data = pickle.load(f)
            embeddings = data["embeddings"]

            self._log_memory("after loading raw embeddings")

            # Ensure embeddings are float32 (PCA outputs float32, no conversion needed)
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)

            self.movie_embeddings = embeddings
            self._log_memory("after loading complete")
            print(f"Embeddings loaded into memory: {self.movie_embeddings.shape}")

        return self.movie_embeddings

    def _log_memory(self, stage=""):
        """Log memory usage if psutil is available"""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / 1024 / 1024
            print(f"[MEMORY] {stage}: {mem_mb:.2f} MB")
        except:
            pass
