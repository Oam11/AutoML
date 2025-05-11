import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
from pycaret.classification import setup as cls_setup, compare_models as cls_compare_models, pull as cls_pull, save_model as cls_save_model
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, pull as reg_pull, save_model as reg_save_model
from pycaret.clustering import setup as clu_setup, create_model as clu_create_model, assign_model as clu_assign_model, pull as clu_pull, save_model as clu_save_model
from pycaret.anomaly import setup as ano_setup, create_model as ano_create_model, assign_model as ano_assign_model, pull as ano_pull, save_model as ano_save_model
from pycaret.time_series import setup as ts_setup, create_model as ts_create_model, predict_model as ts_predict_model, pull as ts_pull, save_model as ts_save_model
from pycaret.datasets import get_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np
from langchain.prompts import PromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
import base64

def initialize_groq_lemma():
    try:
        api_key = st.secrets["groq"]["api_key"]
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
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

    df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
    return df

# Function to preprocess data with PCA
def preprocess_with_pca(df):
    text_columns = df.select_dtypes(include='object').columns
    if text_columns.empty:
        return df

    tfidf_vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_columns].astype(str).agg(' '.join, axis=1))
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(tfidf_matrix.toarray())
    pca_df = pd.DataFrame(pca_result, columns=[f"pca_{i}" for i in range(pca_result.shape[1])])

    df = pd.concat([df.reset_index(drop=True), pca_df], axis=1)
    return df

# Sidebar options
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML with PyCaret")
    choice = st.radio("Navigation", ["Upload Dataset", "Use Sample Dataset", "Exploratory Data Analysis", "Machine Learning", "Ask Questions"])
    st.info("This application allows you to perform automated machine learning tasks using PyCaret and ask questions about your data using Langchain.")

# Load existing data if present
df = None
if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

# Upload Dataset Section
if choice == "Upload Dataset":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload your dataset (CSV format)")
    if file:
        # List of common encodings to try
        encodings = [
            'utf-8',
            'utf-8-sig',
            'latin1',
            'ISO-8859-1',
            'cp1252',
            'ascii',
            'utf-16',
            'utf-16le',
            'utf-16be',
            'utf-32',
            'utf-32le',
            'utf-32be',
            'cp437',
            'cp850',
            'cp1250',
            'cp1251',
            'cp1253',
            'cp1254',
            'cp1255',
            'cp1256',
            'cp1257',
            'cp1258',
            'gbk',
            'gb2312',
            'gb18030',
            'big5',
            'big5hkscs',
            'shift_jis',
            'euc_jp',
            'euc_kr',
            'iso2022_jp',
            'iso2022_kr'
        ]
        
        success = False
        for encoding in encodings:
            try:
                df = pd.read_csv(file, encoding=encoding)
                df.to_csv("sourcedata.csv", index=False, encoding='utf-8')
                st.success(f"Dataset uploaded successfully using {encoding} encoding!")
                st.dataframe(df)
                success = True
                break
            except UnicodeDecodeError:
                continue
            except pd.errors.ParserError as e:
                st.warning(f"Parser error with {encoding} encoding: {str(e)}")
                continue
            except Exception as e:
                st.warning(f"Error with {encoding} encoding: {str(e)}")
                continue
        
        if not success:
            st.error("Failed to read the file with any of the supported encodings. Please check your file format.")

# Use Sample Dataset Section
if choice == "Use Sample Dataset":
    st.title("Select a Sample Dataset")
    dataset_options = {
        "Classification - Iris": ("iris", "classification"),
        "Classification - Diabetes": ("diabetes", "classification"),
        "Classification - Bank": ("bank", "classification"),
        "Regression - Boston": ("boston", "regression"),
        "Regression - Insurance": ("insurance", "regression"),
        "Clustering - Wine": ("wine", "clustering"),
        "Anomaly Detection - Credit Card": ("credit", "anomaly"),
        "Time Series - Airline": ("airline", "time_series")
    }
    selected_dataset = st.selectbox("Choose a dataset", list(dataset_options.keys()))
    if st.button("Load Dataset"):
        dataset_name, task_type = dataset_options[selected_dataset]
        try:
            with st.spinner(f"Loading {selected_dataset} dataset..."):
                data = get_data(dataset_name, verbose=False)
                
                # Handle time series data specifically
                if task_type == "time_series":
                    if isinstance(data, pd.Series):
                        # Convert Series to DataFrame
                        df = pd.DataFrame(data)
                        df.index.name = 'Date'
                        df.columns = ['Passengers']  # Rename the column
                    else:
                        df = data
                else:
                    df = data

                if df is not None and not df.empty:
                    # Save to CSV with appropriate index handling
                    df.to_csv("sourcedata.csv", index=(task_type == "time_series"))
                    st.success(f"{selected_dataset} dataset loaded successfully!")
                    st.write(f"Dataset shape: {df.shape}")
                    st.write("First few rows:")
                    st.dataframe(df.head())
                else:
                    st.error(f"Failed to load {selected_dataset} dataset. The dataset is empty or None.")
        except Exception as e:
            st.error(f"An error occurred while loading the dataset: {str(e)}")
            st.info("""
            Common solutions:
            1. Check your internet connection
            2. Try a different dataset
            3. Make sure you have the latest version of PyCaret installed
            4. If the error persists, try uploading your own dataset instead
            """)

# Exploratory Data Analysis Section
if choice == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    if df is not None:
        profile = ProfileReport(df, title="Profiling Report", explorative=True)
        st_profile_report = profile.to_html()
        # Use full page width and height with a much larger height value
        st.components.v1.html(
            st_profile_report,
            height=3000,  # Much larger height to ensure full page display
            width=None,   # Full width
            scrolling=True
        )
    else:
        st.warning("Please upload or select a dataset first.")

# Machine Learning Section
if choice == "Machine Learning":
    st.title("Automated Machine Learning")
    if df is not None:
        task = st.selectbox("Select the Machine Learning Task", ["Classification", "Regression", "Clustering", "Anomaly Detection", "Time Series Forecasting"])
        if task in ["Classification", "Regression"]:
            target = st.selectbox("Select the Target Column", df.columns)
            embedding_option = st.selectbox("Select Embedding Technique", ["None", "TF-IDF", "PCA"])
            if st.button("Run AutoML"):
                try:
                    df_processed = df.copy()
                    if embedding_option == "TF-IDF":
                        df_processed = preprocess_text_data(df_processed)
                    elif embedding_option == "PCA":
                        df_processed = preprocess_with_pca(df_processed)

                    if task == "Classification":
                        cls_setup(data=df_processed, target=target, session_id=123, verbose=False)
                        best_model = cls_compare_models()
                        st.success("Best Classification Model Trained Successfully!")
                        st.dataframe(cls_pull())
                        cls_save_model(best_model, 'best_classification_model')
                    else:
                        reg_setup(data=df_processed, target=target, session_id=123, verbose=False)
                        best_model = reg_compare_models()
                        st.success("Best Regression Model Trained Successfully!")
                        st.dataframe(reg_pull())
                        reg_save_model(best_model, 'best_regression_model')

                    with open(f"best_{task.lower()}_model.pkl", 'rb') as f:
                        st.download_button(
                            label="Download Trained Model",
                            data=f,
                            file_name=f"best_{task.lower()}_model.pkl",
                            mime="application/octet-stream"
                        )
                except Exception as e:
                    st.error(f"An error occurred during model training: {e}")
        elif task == "Clustering":
            if st.button("Run Clustering"):
                try:
                    clu_setup(data=df, session_id=123, verbose=False)
                    model = clu_create_model('kmeans')
                    df_assigned = clu_assign_model(model)
                    st.success("Clustering Model Trained Successfully!")
                    st.dataframe(clu_pull())
                    clu_save_model(model, 'clustering_model')
                    with open("clustering_model.pkl", 'rb') as f:
                        st.download_button(
                            label="Download Clustering Model",
                            data=f,
                            file_name="clustering_model.pkl",
                            mime="application/octet-stream"
                        )
                except Exception as e:
                    st.error(f"An error occurred during clustering: {e}")
        elif task == "Anomaly Detection":
            if st.button("Run Anomaly Detection"):
                try:
                    ano_setup(data=df, session_id=123, verbose=False)
                    model = ano_create_model('iforest')
                    df_assigned = ano_assign_model(model)
                    st.success("Anomaly Detection Model Trained Successfully!")
                    st.dataframe(ano_pull())
                    ano_save_model(model, 'anomaly_model')
                    with open("anomaly_model.pkl", 'rb') as f:
                        st.download_button(
                            label="Download Anomaly Detection Model",
                            data=f,
                            file_name="anomaly_model.pkl",
                            mime="application/octet-stream"
                        )
                except Exception as e:
                    st.error(f"An error occurred during anomaly detection: {e}")
        elif task == "Time Series Forecasting":
            if df is not None:
                # For sample datasets, we already set the index
                if choice == "Use Sample Dataset" and "airline" in str(selected_dataset):
                    target_column = st.selectbox("Select the Target Column", df.columns)
                    if st.button("Run Time Series Forecasting"):
                        try:
                            # Ensure data is properly formatted for time series
                            if not isinstance(df.index, pd.DatetimeIndex):
                                df.index = pd.to_datetime(df.index)
                            
                            ts_setup(data=df, target=target_column, session_id=123)
                            model = ts_create_model('arima')
                            forecast = ts_predict_model(model)
                            st.success("Time Series Forecasting Model Trained Successfully!")
                            st.dataframe(ts_pull())
                            ts_save_model(model, 'time_series_model')
                            with open("time_series_model.pkl", 'rb') as f:
                                st.download_button(
                                    label="Download Time Series Model",
                                    data=f,
                                    file_name="time_series_model.pkl",
                                    mime="application/octet-stream"
                                )
                        except Exception as e:
                            st.error(f"An error occurred during time series forecasting: {e}")
                else:
                    # For uploaded datasets
                    date_column = st.selectbox("Select the Date Column", df.columns)
                    target_column = st.selectbox("Select the Target Column", [col for col in df.columns if col != date_column])
                    if st.button("Run Time Series Forecasting"):
                        try:
                            df_ts = df[[date_column, target_column]].copy()
                            df_ts[date_column] = pd.to_datetime(df_ts[date_column])
                            df_ts.set_index(date_column, inplace=True)
                            
                            ts_setup(data=df_ts, target=target_column, session_id=123)
                            model = ts_create_model('arima')
                            forecast = ts_predict_model(model)
                            st.success("Time Series Forecasting Model Trained Successfully!")
                            st.dataframe(ts_pull())
                            ts_save_model(model, 'time_series_model')
                            with open("time_series_model.pkl", 'rb') as f:
                                st.download_button(
                                    label="Download Time Series Model",
                                    data=f,
                                    file_name="time_series_model.pkl",
                                    mime="application/octet-stream"
                                )
                        except Exception as e:
                            st.error(f"An error occurred during time series forecasting: {e}")
            else:
                st.warning("Please upload or select a dataset first.")
    else:
        st.warning("Please upload or select a dataset first.")

# Ask Questions Section
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
            st.warning("Groq model initialization failed. Please check your API key in secrets.toml")
