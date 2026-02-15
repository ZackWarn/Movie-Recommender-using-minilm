"use client";
import { useState } from "react";
import { FiSearch } from "react-icons/fi";
import MovieCard from "../components/MovieCard";
import { SkeletonCard } from "./SkeletonCard";
// Poster grid removed per request

const FilmIcon = (props: React.SVGProps<SVGSVGElement>) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="12"      // half of 24
    height="12"     // half of 24
    viewBox="0 0 24 24"  // keep viewbox same for proper scaling
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    {...props}
  >
    <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect>
    <line x1="7" y1="2" x2="7" y2="22"></line>
    <line x1="17" y1="2" x2="17" y2="22"></line>
    <line x1="2" y1="12" x2="22" y2="12"></line>
    <line x1="2" y1="7" x2="7" y2="7"></line>
    <line x1="2" y1="17" x2="7" y2="17"></line>
    <line x1="17" y1="17" x2="22" y2="17"></line>
    <line x1="17" y1="7" x2="22" y2="7"></line>
  </svg>
);

interface MovieType {
  movieId?: number;
  title: string;
  year?: string | number;
  imdb_year?: string | number;
  imdb_rating?: number;
  imdb_rating_count?: number;
  avg_rating?: number;
  poster_url?: string;
  plot?: string;
  genres?: string[];
  imdb_genres?: string[];
  cast?: string[];
  runtime?: number;
  similarity_score?: number;
  imdb_id?: string;
}

export default function CineMatchHero() {
  const [search, setSearch] = useState("");
  const [lastSearchQuery, setLastSearchQuery] = useState("");
  const [recommendations, setRecommendations] = useState<MovieType[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [hasSearched, setHasSearched] = useState(false);
  const MIN_SKELETON_MS = 2000;

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!search.trim()) return;

    setLoading(true);
    setError("");
    setHasSearched(true);
    setRecommendations([]);
    const queryToUse = search.trim();
    const start = performance.now();
    
    try {
      // Call similar movies endpoint instead of semantic query
      const response = await fetch('/api/recommendations/similar', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          movie_id: queryToUse,
          top_k: 8
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch recommendations');
      }

      const data = await response.json();
      setRecommendations(data.recommendations || []);
      setLastSearchQuery(queryToUse); // Save the query that produced these results

      // Removed IMDb fetch
    } catch (err) {
      setError('Failed to get recommendations. Please try again.');
      console.error('Search error:', err);
    } finally {
      const elapsed = performance.now() - start;
      const remaining = Math.max(MIN_SKELETON_MS - elapsed, 0);
      if (remaining > 0) {
        await new Promise((resolve) => setTimeout(resolve, remaining));
      }
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full flex flex-col py-20 bg-gradient-to-br from-[#24033e] via-[#502689] to-[#7423fb] px-4">
      <div className="flex flex-col w-full mx-auto mt-4 px-4 max-w-full">

        <h1 className="text-white text-6xl font-extrabold text-center mb-3 tracking-tight">
          KnowMovies
        </h1>
        <p className="text-purple-100 text-lg text-center mb-7">
          Find movies similar to your favorites. Enter any movie title and discover
          <br />hidden gems you&apos;ll love.
        </p>
        <div className="w-full flex justify-center mb-10">
          <form
            className="relative flex items-center bg-white/10 backdrop-blur-sm rounded-lg shadow w-full max-w-2xl"
            onSubmit={handleSearch}
          >
            {/* FilmIcon positioned at the left end */}
            <FilmIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-purple-200" />
            
            <input
              type="text"
              className="flex-grow py-3 pl-12 pr-20 bg-transparent placeholder-purple-200 outline-none text-amber-100 rounded-xl"
              placeholder="Enter a movie title (e.g., Fight Club, Inception)"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
            <button
              type="submit"
              disabled={loading}
              className="absolute right-1 top-1/2 -translate-y-1/2 flex items-center space-x-2 bg-gradient-to-r from-pink-500 to-purple-600 text-white font-semibold px-6 py-2 rounded-sm transition hover:from-pink-600 hover:to-purple-800 shadow disabled:opacity-50 disabled:cursor-not-allowed"
              style={{height: "2.4rem"}}
            >
              {loading ? (
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              ) : (
                <FiSearch className="h-5 w-5" />
              )}
              <span className="hidden sm:inline">
                {loading ? "Searching..." : "Find Movies"}
              </span>
            </button>
          </form>
        </div>

        {recommendations.length === 0 && !loading && (
          <div className="flex flex-col items-center mt-8 mb-2">
            <span className="text-purple-200 text-6xl mb-4">🎬</span>
            <h2 className="text-white text-2xl font-bold mb-2 text-center">
              Find Movies You'll Love
            </h2>
            <p className="text-purple-200 text-center mb-3">
              Enter a movie title above to discover similar recommendations
            </p>
          </div>
        )}


        {/* Error Message */}
        {error && (
          <div className="mt-8 p-4 bg-red-500/20 border border-red-500/50 rounded-lg">
            <p className="text-red-200 text-center">{error}</p>
          </div>
        )}

        {/* Loading skeletons while waiting for results */}
{loading && (
  <div className="w-full">
    <h2 className="text-white text-3xl font-bold text-center mb-8 animate-pulse">
      Finding movies...
    </h2>
    <div className="grid grid-cols-[repeat(auto-fit,minmax(280px,1fr))] gap-6">
      {Array.from({ length: 8 }).map((_, idx) => (
        <SkeletonCard key={idx} />
        // or <SkeletonCardStaggered key={idx} index={idx} />
      ))}
    </div>
  </div>
)}


        {/* Recommendations Section */}
        {recommendations.length > 0 && !loading && (
  <div className="w-full">
    <h2 className="text-white text-3xl font-bold text-center mb-8">
      Recommendations for &quot;{lastSearchQuery}&quot;
    </h2>
    <div className="grid grid-cols-[repeat(auto-fit,minmax(280px,1fr))] gap-6">
  {recommendations.slice(0, 8).map((movie, index) => (
    <MovieCard key={movie.movieId || index} movie={movie} index={index + 1} />
  ))}
</div>

  </div>
)}

      </div>
    </div>
  );
}
