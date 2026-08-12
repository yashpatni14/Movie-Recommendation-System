import pandas as pd
import re
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Movie Recommendation System")

df = pd.read_csv("tmdb_5000_movies.csv")

# Fill empty text values first.
df["overview"] = df["overview"].fillna("")
df["genres"] = df["genres"].fillna("")
df["keywords"] = df["keywords"].fillna("")

# Use the three text columns as the movie description.
df["clean_text"] = df["overview"] + " " + df["genres"] + " " + df["keywords"]
df["clean_text"] = df["clean_text"].str.lower()
df["clean_text"] = df["clean_text"].apply(lambda x: re.sub(r"[^a-z0-9\s]", " ", x))
df["clean_text"] = df["clean_text"].str.replace(r"\s+", " ", regex=True).str.strip()

stop_words = set("a an the and or but if is are was were of to in on for with from by this that these those it its as at be been being".split())
df["clean_text"] = df["clean_text"].apply(
    lambda x: " ".join(word for word in x.split() if word not in stop_words)
)

# Turn the text into TF-IDF vectors.
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df["clean_text"])
similarity_matrix = cosine_similarity(tfidf_matrix)

def recommend(movie_name, n=5):
    item_index = df.index[df["title"].str.lower() == movie_name.lower()][0]
    scores = similarity_matrix[item_index]
    indexes = scores.argsort()[::-1]
    indexes = indexes[indexes != item_index][:n]

    result = df.iloc[indexes][["title"]].copy()
    result["similarity_score"] = scores[indexes].round(3)
    return result

movies = df["title"].dropna().drop_duplicates().sort_values().tolist()
movie = st.selectbox("Select a movie:", movies)

if st.button("Get Recommendations"):
    st.subheader("Recommended Movies")
    st.dataframe(recommend(movie), width="stretch")
