import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack

def generate_combined_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """
    Combines TF-IDF and LLM-based embeddings into a single feature matrix.
    Args:
        texts (List[str]): List of text documents.
        model_name (str): HuggingFace model name for sentence embeddings.
    Returns:
        combined_features: Sparse/dense matrix of features.
        tfidf_feature_names: Names of TF-IDF features.
    """
    import torch
    # TF-IDF embedding
    tfidf_vectorizer = TfidfVectorizer(max_features=300)  # limit features to avoid memory issues
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    # LLM-based embeddings
    model = SentenceTransformer(model_name)
    llm_embeddings = model.encode(texts, show_progress_bar=True)

    # Combine both
    combined_features = hstack([tfidf_matrix, llm_embeddings])

    return combined_features, tfidf_vectorizer.get_feature_names_out()