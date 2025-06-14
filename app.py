import streamlit as st
from textorbit.data_loader import load_data, select_columns

def main():
    st.set_page_config(page_title="Text Clustering App", layout="wide")
    st.title("📊 Text Data Clustering & Summarization")

    df = load_data()

    if df is not None:
        selected_df = select_columns(df)
        if selected_df is not None:
            st.success("✅ Data and columns are ready to be processed.")
            st.dataframe(selected_df.head())

if __name__ == "__main__":
    main()