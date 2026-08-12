import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("tmdb_5000_movies.csv")

# Fill missing text values
movies["overview"] = movies["overview"].fillna("")
movies["genres"] = movies["genres"].fillna("")
movies["keywords"] = movies["keywords"].fillna("")

# Combine movie text
movies["text"] = (
    movies["overview"] + " " +
    movies["genres"] + " " +
    movies["keywords"]
)

# Convert text to TF-IDF
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies["text"])


def recommend(name_movie):
    movie_index = movies[movies["title"] == name_movie].index[0]

    # Compare only the selected movie with all movies
    recommendations = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    movie_list = sorted(
        enumerate(recommendations),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_list:
        recommended_movies.append({
            "title": movies.iloc[i[0]]["title"],
            "similarity_score": round(i[1], 3)
        })

    return pd.DataFrame(recommended_movies)


# Streamlit interface
st.title("Movie Recommendation System")

movie_name = st.selectbox(
    "Select a movie:",
    movies["title"].tolist()
)

if st.button("Get Recommendations"):
    result = recommend(movie_name)

    st.subheader("Recommended Movies")
    st.dataframe(result, hide_index=True)