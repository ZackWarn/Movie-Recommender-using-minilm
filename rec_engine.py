import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from imdb_service import IMDBService
from config import Config
import logging

logger = logging.getLogger(__name__)


class MovieRecommendationEngine:
    def __init__(self, bert_processor, use_imdb=True):
        self.bert_processor = bert_processor
        # Don't store movies or embeddings directly - access via bert_processor

        # Initialize IMDB service if API key is available
        self.imdb_service = None
        if use_imdb and Config.validate_config():
            try:
                self.imdb_service = IMDBService(Config.RAPIDAPI_IMDB_KEY)
                logger.info("IMDB service initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize IMDB service: {e}")
                self.imdb_service = None

    @property
    def movies(self):
        """Access movies data from bert_processor"""
        return self.bert_processor.movies_data

    def recommend_by_query(self, query, top_k=8):
        """
        Pure semantic similarity-based recommendations using HF Space embeddings.
        """
        logger.info(f"Getting semantic recommendations for query: {query}")

        # Encode query using HF Space
        query_embedding = self.bert_processor.encode([query], force_semantic=True)[0]
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        # Get all movie embeddings
        embeddings = self.bert_processor._get_embeddings()
        embeddings = np.array(embeddings, dtype=np.float32)

        # Compute cosine similarity
        logger.info("Computing semantic similarity scores")
        similarities = cosine_similarity(query_embedding, embeddings)[0]

        # Get top K indices
        top_indices = similarities.argsort()[::-1][:top_k]

        # Build recommendations
        recommendations = []
        for idx in top_indices:
            movie = self.movies.iloc[idx]
            logger.debug(
                "rank=%s title=%s score=%.4f",
                len(recommendations) + 1,
                movie["clean_title"],
                float(similarities[idx]),
            )
            recommendations.append(
                {
                    "movieId": movie["movieId"],
                    "title": movie["clean_title"],
                    "year": movie.get("year", "Unknown"),
                    "genres": movie.get("genres_list", []),
                    "avg_rating": movie.get("avg_rating", 0),
                }
            )

        logger.info(
            f"Semantic ranking complete: returned {len(recommendations)} movies"
        )
        return recommendations

    def search_movies(self, search_term, top_k=20):
        """
        Search movies by matching search_term in cleaned title (case-insensitive).
        Returns up to top_k matching movies with basic info.
        """
        matches = self.movies[
            self.movies["clean_title"].str.contains(search_term, case=False, na=False)
        ]
        results = []
        for _, movie in matches.head(top_k).iterrows():
            results.append(
                {
                    "movieId": movie["movieId"],
                    "title": movie["clean_title"],
                    "year": movie.get("year", "Unknown"),
                    "genres": movie.get("genres_list", []),
                    "avg_rating": movie.get("avg_rating", 0),
                }
            )
        return results

    def recommend_similar_movies(self, movie_id, top_k=8):
        """
        Find movies similar to the given movie_id using cosine similarity.

        Args:
            movie_id: The ID of the movie to find similar movies for
            top_k: Number of recommendations to return (default 8)

        Returns:
            List of similar movie recommendations with scores
        """
        logger.info(f"Finding movies similar to movie_id: {movie_id}")

        # Find the movie in the dataset
        movie_matches = self.movies[self.movies["movieId"] == int(movie_id)]
        if movie_matches.empty:
            logger.warning(f"Movie ID {movie_id} not found")
            return []

        movie_idx = movie_matches.index[0]
        movie = movie_matches.iloc[0]
        logger.info(
            f"Found movie: {movie['clean_title']} ({movie.get('year', 'Unknown')})"
        )

        # Get all movie embeddings
        embeddings = self.bert_processor._get_embeddings()
        embeddings = np.array(embeddings, dtype=np.float32)

        # Get the embedding for the target movie
        movie_embedding = embeddings[movie_idx].reshape(1, -1)

        # Compute cosine similarity with all movies
        logger.info("Computing similarity scores")
        similarities = cosine_similarity(movie_embedding, embeddings)[0]

        # Get top K+1 indices (excluding the movie itself)
        top_indices = similarities.argsort()[::-1][: top_k + 1]

        # Build recommendations, excluding the input movie
        recommendations = []
        for idx in top_indices:
            if idx == movie_idx:
                continue  # Skip the input movie itself

            similar_movie = self.movies.iloc[idx]
            logger.debug(
                "rank=%s similar_to=%s candidate=%s score=%.4f",
                len(recommendations) + 1,
                movie["clean_title"],
                similar_movie["clean_title"],
                float(similarities[idx]),
            )
            recommendations.append(
                {
                    "movieId": similar_movie["movieId"],
                    "title": similar_movie["clean_title"],
                    "year": similar_movie.get("year", "Unknown"),
                    "genres": similar_movie.get("genres_list", []),
                        "avg_rating": similar_movie.get("avg_rating", 0),
                }
            )

            if len(recommendations) >= top_k:
                break

        logger.info(f"Found {len(recommendations)} similar movies")
        return recommendations

    def _enhance_with_imdb_data(self, recommendations):
        """Enhance recommendations with IMDB data if available"""
        if not self.imdb_service:
            return recommendations

        enhanced_recommendations = []
        for rec in recommendations:
            enhanced_rec = rec.copy()

            try:
                # Search for the movie in IMDB
                imdb_results = self.imdb_service.search_movies(rec["title"], limit=1)
                if imdb_results:
                    # Pick best-matching IMDb result to avoid wrong first hits like "De små mænd"
                    def score_candidate(c):
                        title_c = str(c.get("title", "") or "").lower()
                        title_q = str(rec["title"]).lower()
                        score = 0.0
                        if title_c == title_q:
                            score += 3.0
                        elif title_c.startswith(title_q) or title_q.startswith(title_c):
                            score += 1.5
                        elif title_q in title_c or title_c in title_q:
                            score += 0.8
                        # Prefer same year if available
                        try:
                            yr_q = int(rec.get("year") or 0)
                            yr_c = int(c.get("year") or 0)
                            if yr_q and yr_c and yr_q == yr_c:
                                score += 0.7
                        except Exception:
                            pass
                        # Prefer items with poster
                        if c.get("poster_url") or c.get("image_url"):
                            score += 0.3
                        # Prefer movies over TV
                        t = (c.get("type") or "").lower()
                        if t in ("movie", "tvmovie"):
                            score += 0.4
                        return score

                    imdb_movie = max(imdb_results, key=score_candidate)
                    enhanced_rec.update(
                        {
                            "imdb_id": imdb_movie.get("imdb_id", ""),
                            "imdb_rating": imdb_movie.get("rating", 0),
                            "imdb_rating_count": imdb_movie.get("rating_count", 0),
                            "poster_url": imdb_movie.get("poster_url")
                            or imdb_movie.get("image_url", ""),
                            "imdb_year": imdb_movie.get("year", ""),
                            "actors": imdb_movie.get("actors", ""),
                            "imdb_rank": imdb_movie.get("rank", 0),
                        }
                    )
                    # Avoid imdb236 detail endpoints (often 404); rely on search fields for poster
            except Exception as e:
                logger.warning(
                    f"Failed to enhance movie {rec['title']} with IMDB data: {e}"
                )

            enhanced_recommendations.append(enhanced_rec)

        return enhanced_recommendations

    def recommend_by_query_with_imdb(self, query, top_k=10):
        """Get recommendations with IMDB data enhancement"""
        recommendations = self.recommend_by_query(query, top_k)
        return self._enhance_with_imdb_data(recommendations)

    def recommend_similar_movies_with_imdb(self, movie_id, top_k=8):
        """Get similar movies with IMDB data enhancement"""
        recommendations = self.recommend_similar_movies(movie_id, top_k)
        return self._enhance_with_imdb_data(recommendations)

    def search_movies_with_imdb(self, search_term, top_k=20):
        """Search movies with IMDB data enhancement"""
        recommendations = self.search_movies(search_term, top_k)
        return self._enhance_with_imdb_data(recommendations)

    def get_trending_movies(self, limit=10):
        """Get trending movies from IMDB"""
        if not self.imdb_service:
            logger.warning("IMDB service not available")
            return []

        try:
            return self.imdb_service.get_trending_movies(limit)
        except Exception as e:
            logger.error(f"Failed to get trending movies: {e}")
            return []

    def search_imdb_movies(self, query, limit=10):
        """Search movies directly from IMDB"""
        if not self.imdb_service:
            logger.warning("IMDB service not available")
            return []

        try:
            return self.imdb_service.search_and_get_details(query, limit)
        except Exception as e:
            logger.error(f"Failed to search IMDB movies: {e}")
            return []
