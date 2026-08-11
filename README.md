# Movie Recommendation System

This project follows the uploaded assignment structure and keeps Tasks 1-5 in a simple Jupyter Notebook.

## Dataset

Kaggle: TMDB 5000 Movie Dataset

https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

The dataset contains movie metadata including title, overview, genres and keywords.

## Files

- `Movie_Recommendation_Tasks_1_to_5.ipynb` — Tasks 1-5
- `app.py` — Task 6 Streamlit interface
- `requirements.txt` — required Python packages
- `README.md` — project information

Place `tmdb_5000_movies.csv` in the same folder as the notebook and `app.py`.

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
Create a Git repository and push:
- `app.py`
- `requirements.txt`
- `README.md`
- the notebook
- dataset file only if your course/project rules allow it

### Task 8 — Render
Create a Render Web Service connected to GitHub.
Build command:
`pip install -r requirements.txt`

Start command:
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

### Task 9 — Final Validation
Open the deployed URL, select a movie, click the recommendation button, and verify that recommendations are displayed.
