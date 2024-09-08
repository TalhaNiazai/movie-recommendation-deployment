from flask import Flask, jsonify, request
import redis
from database import load_movies
from scipy.sparse import load_npz
import logging
import pickle
import os
import requests  # Add requests to download the matrix file
from dotenv import load_dotenv

app = Flask(__name__)

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Redis connection setup using environment variables
redis_client = redis.StrictRedis(
    host=os.getenv('REDIS_HOST'),
    port=int(os.getenv('REDIS_PORT')),
    password=os.getenv('REDIS_PASSWORD'),
    decode_responses=False  # Ensure binary responses
)

# GitHub release URL for the similarity matrix
MATRIX_URL = 'https://github.com/TalhaNiazai/movie-recommendation-deployment/releases/download/v1.0/similarity_matrix.npz'

# Path to save the downloaded file
MATRIX_FILE_PATH = 'similarity_matrix.npz'

# Function to download the file if it doesn't exist locally


def download_similarity_matrix():
    if not os.path.exists(MATRIX_FILE_PATH):
        logging.info("Downloading similarity matrix from GitHub...")
        response = requests.get(MATRIX_URL)
        with open(MATRIX_FILE_PATH, 'wb') as f:
            f.write(response.content)
        logging.info("Download complete.")

# Download the matrix if necessary
download_similarity_matrix()

# Load similarity matrix
with open(MATRIX_FILE_PATH, 'rb') as f:
    similarity = load_npz(f)

# Load the movie list from PostgreSQL
movies = load_movies()

@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        movie_title = request.args.get('movie')

        if not movie_title:
            return jsonify({"error": "Movie title is required"}), 400

        # Check if recommendations are cached in Redis
        cached_recommendations = redis_client.get(movie_title)
        if cached_recommendations:
            try:
                recommendations = pickle.loads(cached_recommendations)
                return jsonify(recommendations=recommendations)
            except pickle.PickleError as e:
                logging.error(f"Error decoding cached recommendations: {e}")
                return jsonify({"error": "Error decoding cached recommendations"}), 500

        # Check if the movie is in the dataset
        if movie_title in movies['title'].values:
            index = movies[movies['title'] == movie_title].index[0]
            distances = similarity[index].toarray().flatten()
            distance_with_index = list(enumerate(distances))
            distance_with_index.sort(key=lambda x: x[1], reverse=True)

            # Get top 5 similar movies
            recommendations = []
            for i in distance_with_index[1:6]:
                recommend_movie = movies.iloc[i[0]].title
                recommendations.append(recommend_movie)

            # Cache the recommendations in Redis
            try:
                redis_client.set(movie_title, pickle.dumps(recommendations), ex=86400)  # Cache for 24 hours
            except pickle.PickleError:
                return jsonify({"error": "Error encoding recommendations for cache"}), 500

            return jsonify(recommendations=recommendations)

        else:
            return jsonify({"error": "Movie not found"}), 404

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)
