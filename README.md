# Movie Recommendation System

## Dataset

Kaggle: TMDB 5000 Movie Dataset

https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

The dataset contains movie metadata including title, overview, genres and keywords.

## Project Links

### GitHub Repository

https://github.com/yashpatni14/Movie-Recommendation-System

### Live Render Application

https://movie-recommendation-system-etrf.onrender.com

## Files

- `Movie_Recommendation.ipynb` — Tasks 1-5
- `app.py` — Task 6 Streamlit interface
- `requirements.txt` — required Python packages
- `tmdb_5000_movies.csv` — movie dataset
- `README.md` — project information and links

## Tasks 1-5

1. Load and understand the dataset.
2. Preprocess the movie text.
3. Convert text into TF-IDF vectors.
4. Compute cosine similarity.
5. Build and test a content-based recommendation function.

## Other Tasks

### Task 6 — Streamlit

Use `app.py` for the simple dropdown + recommendation button interface.

Run:

`streamlit run app.py`

### Task 7 — Git & GitHub

The project is available on GitHub:

https://github.com/yashpatni14/Movie-Recommendation-System

### Task 8 — Render

The application is deployed on Render.

Build command:

`pip install -r requirements.txt`

Start command:

`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

### Task 9 — Final Validation

Live application:

https://movie-recommendation-system-etrf.onrender.com