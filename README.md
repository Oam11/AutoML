# AutoML 

A powerful web application that combines automated machine learning capabilities with interactive data analysis and natural language processing. Built with Streamlit and PyCaret, this application makes machine learning accessible to everyone.

## Features

### 1. Data Management
- Upload your own CSV datasets with automatic encoding detection
- Support for 30+ different file encodings including UTF-8, Latin1, and various Asian encodings
- Choose from pre-loaded sample datasets
- Automatic data persistence between sessions

### 2. Exploratory Data Analysis
- Interactive data profiling with ydata-profiling
- Full-page report display with comprehensive statistics
- Visual data exploration
- Automatic data type detection and analysis

### 3. Automated Machine Learning
- **Classification**
  - Multiple model comparison
  - Model performance metrics
  - Model download capability
- **Regression**
  - Multiple model comparison
  - Model performance metrics
  - Model download capability
- **Clustering**
  - K-means clustering
  - Cluster assignment visualization
- **Anomaly Detection**
  - Isolation Forest implementation
  - Anomaly scoring
- **Time Series Forecasting**
  - ARIMA modeling
  - Time series visualization
  - Forecast predictions

### 4. Advanced Preprocessing
- TF-IDF text preprocessing with customizable features
- PCA dimensionality reduction
- Automatic feature engineering
- Support for mixed data types
- Text data transformation

### 5. Natural Language Interface
- Powered by Groq's Llama 3.3 70B model
- Interactive data exploration through natural language
- Intelligent data analysis responses
- Context-aware question answering

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AutoML.git
cd AutoML
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.streamlit/secrets.toml` file with your Groq API key:
```toml
[groq]
api_key = "your-groq-api-key"
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Navigate through the different sections using the sidebar:
   - **Upload Dataset**: Import your own CSV files with automatic encoding detection
   - **Use Sample Dataset**: Choose from pre-loaded datasets
   - **Exploratory Data Analysis**: View detailed data profiles
   - **Machine Learning**: Run automated ML tasks
   - **Ask Questions**: Interact with your data using natural language

## Supported Sample Datasets

### Classification
- Iris: Classic classification dataset
- Diabetes: Medical classification dataset
- Bank: Banking customer classification

### Regression
- Boston: Housing price prediction
- Insurance: Insurance cost prediction

### Clustering
- Wine: Wine quality clustering

### Anomaly Detection
- Credit Card: Credit card fraud detection

### Time Series
- Airline: Airline passenger forecasting

## Model Features

### Classification & Regression
- Multiple model comparison
- Performance metrics visualization
- Model download capability
- Feature importance analysis
- Optional preprocessing with TF-IDF or PCA

### Clustering
- K-means implementation
- Cluster visualization
- Cluster assignment export

### Anomaly Detection
- Isolation Forest algorithm
- Anomaly score calculation
- Threshold-based detection

### Time Series
- ARIMA modeling
- Forecast visualization
- Time series decomposition

## Requirements

The project uses the following main packages:
- streamlit
- pycaret
- ydata-profiling
- langchain
- groq
- pandas
- numpy
- scikit-learn
- plotly
- matplotlib
- seaborn

For a complete list of dependencies, see `requirements.txt`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. When contributing:
1. Fork the repository
2. Create a new branch for your feature
3. Submit a pull request with a clear description of your changes

## License

This project is licensed under the MIT License - see the LICENSE file for details.
