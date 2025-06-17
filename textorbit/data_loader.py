import streamlit as st
import pandas as pd

def load_data():
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            df = df.sample(100)
            st.write("Disclaimer: A Sample of 100 Data is extracted from input!")
            st.subheader("📄 Uploaded Data Preview")
            st.dataframe(df.head())
            return df
        except Exception as e:
            st.error(f"Failed to load file: {e}")
    else:
        st.info("Please upload a file to continue.")
    return None

def select_columns(df):
    with st.form("column_form"):
        st.subheader("🧩 Select Columns")
        id_col = st.selectbox("Select ID column", df.columns)
        text_col = st.selectbox("Select Text column", df.columns)
        submit = st.form_submit_button("Confirm")

    if submit:
        selected_df = df[[id_col, text_col]].dropna().copy()
        selected_df.columns = ["id", "text"]  # normalize column names
        st.session_state["selected_df"] = selected_df
        return selected_df
    return None