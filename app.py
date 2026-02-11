import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# ===================== TITLE =====================
st.title("Hierarchical Clustering on Financial News")

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    df1 = pd.read_csv(
        "all_data.csv",
        encoding='latin1',
        header=None
    )

    df1.columns = ['Sentiment', 'News']
    newsdf = df1['News'].to_frame()
    return newsdf


newsdf = load_data()

st.write("Total Articles:", newsdf.shape[0])

# ===================== TF–IDF =====================
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=1000
)

X_tfidf = vectorizer.fit_transform(newsdf['News'])

st.write("TF-IDF Matrix Shape:", X_tfidf.shape)

# ===================== DENDROGRAM =====================
st.subheader("Dendrogram (First 100 Articles)")

subset = X_tfidf[:100].toarray()
Z = linkage(subset, method='ward')

fig = plt.figure(figsize=(12, 6))

dendrogram(Z)
plt.axhline(y=10, color='r', linestyle='--')

plt.title("News Articles Dendrogram")
plt.xlabel("Articles")
plt.ylabel("Distance")

st.pyplot(fig)

# ===================== SELECT CLUSTERS =====================
k = st.slider("Select Number of Clusters", 2, 10, 5)

# ===================== HIERARCHICAL CLUSTERING =====================
model = AgglomerativeClustering(
    n_clusters=k,
    linkage='ward'
)

clusters = model.fit_predict(X_tfidf.toarray())

newsdf['Cluster'] = clusters

st.write("### Clustered Data Sample")
st.write(newsdf.head())

# ===================== VIEW BY CLUSTER =====================
st.subheader("View Articles by Cluster")

selected = st.selectbox(
    "Choose Cluster",
    sorted(newsdf['Cluster'].unique())
)

filtered = newsdf[newsdf['Cluster'] == selected]

st.write(filtered['News'].head(10))

# ===================== SILHOUETTE SCORE =====================
score = silhouette_score(X_tfidf, clusters)

st.write("### Silhouette Score:", round(score, 4))

# ===================== DOWNLOAD RESULT =====================
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


csv = convert_df(newsdf)

st.download_button(
    "Download Clustered CSV",
    csv,
    "clustered_news.csv",
    "text/csv"
)
