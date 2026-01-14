# data_processor.py
import pandas as pd
import numpy as np
from collections import Counter
import re


def normalize_title(title):
    """Convert titles like 'Dark Knight Rises, The' to 'The Dark Knight Rises'"""
    # First remove year if present
    title_without_year = re.sub(r"\s*\(\d{4}\)$", "", title)

    # Check if title ends with ", The", ", A", or ", An"
    article_pattern = r"^(.+),\s+(The|A|An)$"
    match = re.match(article_pattern, title_without_year, re.IGNORECASE)

    if match:
        main_title = match.group(1)
        article = match.group(2)
        return f"{article} {main_title}"

    return title_without_year


def load_and_preprocess_data():
    # Load datasets
    movies = pd.read_csv("movies_dataset/movies.csv")
    ratings = pd.read_csv("movies_dataset/ratings.csv")
    tags = pd.read_csv("movies_dataset/tags.csv")
    genome_scores = pd.read_csv("movies_dataset/genome-scores.csv")
    genome_tags = pd.read_csv("movies_dataset/genome-tags.csv")

    # Clean movie titles and extract years
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$")
    movies["clean_title"] = movies["title"].apply(normalize_title)

    # Process genres
    movies["genres_list"] = movies["genres"].str.split("|")

    # Calculate movie statistics from ratings
    movie_stats = ratings.groupby("movieId").agg({"rating": ["mean", "count"]}).round(2)
    movie_stats.columns = ["avg_rating", "rating_count"]

    # Merge all movies with rating stats (no popularity filter)
    movies_final = movies.merge(
        movie_stats, left_on="movieId", right_index=True, how="left"
    )
    # Fill missing stats for rarely-rated movies
    movies_final["avg_rating"] = movies_final["avg_rating"].fillna(0)
    movies_final["rating_count"] = movies_final["rating_count"].fillna(0).astype(int)

    return movies_final, ratings, tags, genome_scores, genome_tags


def prepare_movie_tags(movies, tags, genome_scores, genome_tags):
    # Get top user tags for each movie
    movie_tags = tags.groupby("movieId")["tag"].apply(list).to_dict()

    # Get relevant genome tags (relevance > 0.5)
    high_relevance = genome_scores[genome_scores["relevance"] > 0.5]

    # Merge with tag names
    genome_with_names = high_relevance.merge(genome_tags, on="tagId")

    # Get top genome tags per movie
    genome_tags_per_movie = (
        genome_with_names.groupby("movieId")
        .apply(lambda x: x.nlargest(10, "relevance")["tag"].tolist())
        .to_dict()
    )

    # Combine user tags and genome tags
    def get_combined_tags(movie_id):
        user_tags = movie_tags.get(movie_id, [])
        genome_tags = genome_tags_per_movie.get(movie_id, [])

        # Combine and get most frequent
        all_tags = user_tags + genome_tags
        tag_counts = Counter(all_tags)
        return [tag for tag, count in tag_counts.most_common(15)]

    movies["combined_tags"] = movies["movieId"].apply(get_combined_tags)

    return movies
