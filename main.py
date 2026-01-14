# main.py
from data_prep import load_and_preprocess_data, prepare_movie_tags
from bert_processor import MovieBERTProcessor
from rec_engine import MovieRecommendationEngine


def main():
    # Step 1: Load and preprocess data
    print("Loading data...")
    movies, ratings, tags, genome_scores, genome_tags = load_and_preprocess_data()

    # Step 2: Prepare movie tags
    print("Preparing movie tags...")
    movies = prepare_movie_tags(movies, tags, genome_scores, genome_tags)

    # Step 3: Initialize BERT processor
    print("Initializing BERT model...")
    bert_processor = MovieBERTProcessor()

    # Step 4: Generate embeddings
    print("Generating embeddings...")
    bert_processor.generate_embeddings(movies)

    # Step 5: Save embeddings for future use
    bert_processor.save_embeddings()

    # Step 6: Create recommendation engine
    engine = MovieRecommendationEngine(bert_processor)

    # Test the system
    print("\n" + "=" * 50)
    print("TESTING THE RECOMMENDATION SYSTEM")
    print("=" * 50)

    # Test 1: Natural language query
    print("\nTest 1: Natural language query")
    query = "action movies with time travel and sci-fi elements"
    recommendations = engine.recommend_by_query(query, top_k=5)

    print(f"\nRecommendations for: '{query}'")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} ({rec['year']})")
        print(f"   Genres: {', '.join(rec['genres'])}")
        explanation = rec.get("explanation")
        if explanation:
            print(f"   {explanation}")

    # Test 2: Similar movies
    print("\nTest 2: Find similar movies")
    # Find a popular sci-fi movie
    sci_fi_movies = movies[movies["genres"].str.contains("Sci-Fi", na=False)]
    test_movie = sci_fi_movies.iloc[0]

    similar_movies = engine.recommend_similar_movies(test_movie["movieId"], top_k=5)
    print(f"\nMovies similar to '{test_movie['clean_title']}':")
    for i, rec in enumerate(similar_movies, 1):
        print(f"{i}. {rec['title']} ({rec['year']})")
        print(f"   Genres: {', '.join(rec['genres'])}")


if __name__ == "__main__":
    main()
