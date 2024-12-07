import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model
from pandas.api.types import is_numeric_dtype
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np
from langchain.prompts import PromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
import pdfkit

# Ensure tabulate is installed
try:
    import tabulate
except ImportError:
    st.error("Missing optional dependency 'tabulate'. Please install it using 'pip install tabulate'.")
    st.stop()

def initialize_groq_lemma():
    try:
        api_key = st.secrets["groq"]["api_key"]
        llm = ChatGroq(
            model="llama-3.2-90b-text-preview",
            temperature=0.2,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=api_key
        )
        return llm
    except Exception as e:
        st.error(f"An error occurred while initializing model with Groq API: {str(e)}")
        return None

# Function to preprocess text data with TF-IDF
def preprocess_text_data(df):
    text_columns = df.select_dtypes(include='object').columns
    if text_columns.empty:
        return df
    
    tfidf_vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_columns].astype(str).agg(' '.join, axis=1))
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
    
    # Adding prefix to column names to avoid conflicts
    df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
    return df

# Function to preprocess data with PCA
def preprocess_with_pca(df):
    text_columns = df.select_dtypes(include='object').columns
    if text_columns.empty:
        return df

    tfidf_vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_columns].astype(str).agg(' '.join, axis=1))
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    pca_result = pca.fit_transform(tfidf_matrix.toarray())
    pca_df = pd.DataFrame(pca_result, columns=[f"pca_{i}" for i in range(pca_result.shape[1])])
    
    # Adding prefix to column names to avoid conflicts
    df = pd.concat([df.reset_index(drop=True), pca_df], axis=1)
    return df

# Sidebar options
with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML with Langchain")
    choice = st.radio("Navigation", ["Upload Your Dataset", "Exploratory Data Analysis", "Machine Learning", "Ask Questions"])
    st.info("This project helps you to Automate Machine Learning, Data Analysis, and Ask Questions Using Langchain.")

# Load existing data if present
df = None
if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

# Upload Dataset Section
if choice == "Upload Your Dataset":
    st.title("Upload Your Data for Modelling!")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                df = pd.read_csv(file, index_col=None, encoding=encoding)
                df.to_csv("sourcedata.csv", index=None)
                st.dataframe(df)
                break
            except UnicodeDecodeError:
                st.warning(f"UnicodeDecodeError with encoding: {encoding}. Trying next encoding...")
                continue
            except pd.errors.ParserError as e:
                st.error(f"Parsing error with encoding {encoding}: {str(e)}. Trying next encoding...")
                continue
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                break

# EDA Section
if choice == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    if df is not None:
        profile_report = ProfileReport(df)
        st_profile_report(profile_report)
        
        # Save the profile report to HTML
        profile_html = profile_report.to_html()
        st.download_button(
            label="Download Profile Report (HTML)",
            data=profile_html,
            file_name="profile_report.html",
            mime="text/html"
        )
    else:
        st.warning("Please upload a dataset first.")

# Machine Learning Section
if choice == "Machine Learning":
    st.title("All Machine Learning Models")
    if df is not None:
        target = st.selectbox("Select Your Target", df.columns)
        
        embedding_option = st.selectbox("Select Embedding Technique", ["None", "TF-IDF", "PCA"])
        if st.button("Apply Embedding"):
            try:
                if embedding_option == "TF-IDF":
                    df = preprocess_text_data(df)
                elif embedding_option == "PCA":
                    df = preprocess_with_pca(df)
                
                # Set up PyCaret without the 'silent' parameter
                setup_df = setup(df, target=target, session_id=123)
                st.info("This is the ML Experiment settings")
                st.dataframe(pull(), width=800, height=600)
                
                # Compare models
                best_model = compare_models()
                st.info("This is the ML Model")
                st.dataframe(pull())
                
                # Save the best model
                save_model(best_model, 'best_model')
                
                st.success("Model trained and saved successfully.")
                
                # Add download button for the trained model
                if os.path.exists("best_model.pkl"):
                    with open("best_model.pkl", 'rb') as f:
                        st.download_button(
                            label="Download the Model",
                            data=f.read(),
                            file_name="trained_model.pkl",
                            mime="application/octet-stream"
                        )
            except Exception as e:
                st.error(f"An error occurred during model training: {str(e)}")
    else:
        st.warning("Please upload a dataset first.")

# Ask Questions Section using Langchain and Groq Mixtral
if choice == "Ask Questions":
    st.title("Ask Questions About Your Data")
    question = st.text_input("Enter your question:")
    
    llm = initialize_groq_lemma()

    if llm and question and df is not None:
        if st.button("Ask"):
            try:
                agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
                answer = agent.run(question)
                st.write("Answer:", answer)
            except Exception as e:
                st.error(f"An error occurred while processing your question: {str(e)}")
    else:
        if df is None:
            st.warning("Please upload a dataset first.")
        elif llm is None:
            st.warning("Groq model initialization failed.")
