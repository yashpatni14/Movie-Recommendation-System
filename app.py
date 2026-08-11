import pandas as pd
import re
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

st.title("Movie Recommendation System")

df = pd.read_csv("tmdb_5000_movies.csv")

df["overview"] = df["overview"].fillna("")
df["genres"] = df["genres"].fillna("")
df["keywords"] = df["keywords"].fillna("")

df["clean_text"] = (
    df["overview"] + " " +
    df["genres"] + " " +
    df["keywords"]
)

df["clean_text"] = df["clean_text"].str.lower()
df["clean_text"] = df["clean_text"].apply(
    lambda x: re.sub(r"[^a-z0-9\s]", " ", x)
)
df["clean_text"] = df["clean_text"].str.replace(
    r"\s+", " ", regex=True
).str.strip()

df["clean_text"] = df["clean_text"].apply(
    lambda x: " ".join(
        word for word in x.split()
        if word not in ENGLISH_STOP_WORDS
    )
)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df["clean_text"])
similarity_matrix = cosine_similarity(tfidf_matrix)

def recommend(item_name, top_n=5):
    matches = df.index[
        df["title"].str.lower() == item_name.lower()
    ].tolist()

    if not matches:
        return pd.DataFrame(columns=["title", "similarity_score"])

    item_index = matches[0]
    scores = list(enumerate(similarity_matrix[item_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [item for item in scores if item[0] != item_index]

    top_items = scores[:top_n]

    return pd.DataFrame({
        "title": [df.iloc[i]["title"] for i, score in top_items],
        "similarity_score": [round(score, 3) for i, score in top_items]
    })

movie = st.selectbox(
    "Select a movie:",
    df["title"].dropna().drop_duplicates().sort_values().tolist()
)

if st.button("Get Recommendations"):
    st.subheader("Recommended Movies")
    st.dataframe(recommend(movie, top_n=5), use_container_width=True)
