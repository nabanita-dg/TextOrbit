import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.sparse import csr_matrix
from collections import defaultdict

def cluster_texts(X, ids, tfidf_features, num_clusters=5):
    """
    Perform clustering and return cluster-level summary.
    Args:
        X: Combined feature matrix (sparse or dense).
        ids: List of document IDs (e.g., feedback_id).
        tfidf_features: List of TF-IDF feature names.
        num_clusters: Number of clusters.
    Returns:
        cluster_df: DataFrame with cluster_id, text_ids, score, keywords.
        labels: Cluster labels per document.
    """
    km = KMeans(n_clusters=num_clusters, random_state=42)
    labels = km.fit_predict(X)

    cluster_text_ids = defaultdict(list)
    cluster_scores = {}
    cluster_keywords = {}
    num_of_texts = []

    # Inverse TF-IDF part of X (sparse) to extract top keywords
    X = csr_matrix(X)
    tfidf_part = X[:, :len(tfidf_features)].toarray()  # assuming first N columns are TF-IDF

    for i in range(num_clusters):
        indices = np.where(labels == i)[0]
        cluster_text_ids[i] = [ids[idx] for idx in indices]
        num_of_texts.append(len(cluster_text_ids[i]))

        # Score: Mean distance to centroid (cohesiveness)
        _, distances = pairwise_distances_argmin_min(km.cluster_centers_[i].reshape(1, -1), X[indices])
        cluster_scores[i] = round(float(np.mean(distances)), 4)

        # Top keywords: Average TF-IDF score
        tfidf_avg = np.mean(tfidf_part[indices], axis=0)
        top_keyword_indices = tfidf_avg.argsort()[::-1][:5]
        top_keywords = [tfidf_features[idx] for idx in top_keyword_indices]
        cluster_keywords[i] = top_keywords

    cluster_df = pd.DataFrame({
        "Cluster ID": list(cluster_text_ids.keys()),
        "Text IDs": list(cluster_text_ids.values()),
        "Number of Text IDs": num_of_texts,
        "Score": [cluster_scores[i] for i in cluster_text_ids.keys()],
        "Keywords": [", ".join(cluster_keywords[i]) for i in cluster_keywords.keys()]
    })

    return cluster_df, labels