import streamlit as st
from textorbit.data_loader import load_data, select_columns
from textorbit.embedding_generator import generate_combined_embeddings
from textorbit.cluster_engine import cluster_texts
from textorbit.cluster_summary import generate_cluster_summary
from huggingface_hub import login
import openai

import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file
login(os.getenv("HUGGINGFACE_TOKEN"))
#openai.api_key = os.getenv("OPENAI_API_KEY")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----------------- Load Data -----------------
def load_and_select_data():
    df = load_data()  # Your custom file upload and CSV read
    if df is not None:
        selected_df = select_columns(df)  # User selects "text" and "id" columns
        return selected_df
    return None

# ----------------- Generate Embeddings -----------------
@st.cache_data
def get_embeddings(texts):
    return generate_combined_embeddings(texts)

# ----------------- Perform Clustering ------------------
@st.cache_data
def get_clustered_data(texts, ids, tfidf_features, _embeddings):
    return cluster_texts(_embeddings, ids, tfidf_features, num_clusters=5)

# ----------------- UI Section -----------------
def show_cluster_summary(cluster_df):
    st.subheader("📌 Cluster Summary")
    st.dataframe(cluster_df)

def show_cluster_explorer(cluster_df, selected_df):
    st.subheader("🔍 Explore Cluster Details")

    cluster_ids = sorted(cluster_df["Cluster ID"].unique().tolist())

    selected_cluster_id = st.selectbox(
        "Select a Cluster ID to view its contents",
        options=["-- Select Cluster ID --"] + [str(cid) for cid in cluster_ids],
        index=0,
    )

    if selected_cluster_id != "-- Select Cluster ID --":
        cluster_id = int(selected_cluster_id)
        get_cluster_details(cluster_df, selected_df, cluster_id)
    else:
        st.info("Please select a cluster ID to view its details.")

def get_cluster_details(cluster_df, selected_df, cluster_id_selected):
    '''# Display keywords and score
    st.markdown(f"**🧠 Keywords:** {cluster_row['Keywords']}")
    st.markdown(f"**📈 Cluster Score:** {cluster_row['Score']}")'''
    # Get the cluster row
    cluster_row = cluster_df[cluster_df["Cluster ID"] == cluster_id_selected].iloc[0]
    text_ids = cluster_row["Text IDs"]
    cluster_data = selected_df[selected_df["id"].isin(text_ids)]

    # Use columns layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"### 🧾 Texts in Cluster {cluster_id_selected}")
        st.dataframe(cluster_data[["id", "text"]])

    with col2:
        st.markdown(f"### 🧠 LLM Summary for Cluster {cluster_id_selected}")
        with st.spinner("Generating summary with LLM..."):
            summary, elapsed_time = generate_cluster_summary(cluster_data["text"].tolist())
            st.markdown(summary)
            st.caption(f"⏱️ Time taken: {elapsed_time:.2f} seconds")

# ----------------- Main Function -----------------
def main():
    st.set_page_config(page_title="Text Clustering App", layout="wide")
    st.title("📊 Text Data Clustering & Summarization")
    if "selected_df" not in st.session_state:
        df = load_data()
        if df is not None:
            selected = select_columns(df)
            if selected is not None:
                st.session_state.selected_df = selected

    if "selected_df" in st.session_state:
        selected_df = st.session_state.selected_df
        st.success("✅ Data and columns are ready to be processed.")
        texts = selected_df["text"].tolist()
        ids = selected_df["id"].tolist()

        embeddings, tfidf_features = get_embeddings(texts)
        cluster_df, labels = get_clustered_data(texts, ids, tfidf_features, embeddings)

        show_cluster_summary(cluster_df)

        show_cluster_explorer(cluster_df, selected_df)



if __name__ == "__main__":
    main()