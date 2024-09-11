
Welcome to **AutoML**! This web application is designed to simplify and automate your data analysis and machine learning workflows. Whether you're a data scientist, analyst, or just getting started, this tool provides an intuitive interface for exploring and modeling your data.

**Setup**
1. Clone the repository:

   ```bash
   git clone https://github.com/oam11/automl.git
   cd automl
   ```

2. Set up the environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure your API keys in `secrets.toml`:

   ```toml
   [groq]
   api_key = "YOUR_GROQ_API_KEY"
   ```

 **Usage**

1. **Run the Application**:

   ```bash
   streamlit run app.py
   ```

2. **Navigate Through the Application**:
   - **Upload Your Dataset**: Upload your CSV file and view the data.
   - **Exploratory Data Analysis**: Analyze your data and download the profile report.
   - **Machine Learning**: Select your target variable, apply preprocessing, and train machine learning models.
   - **Ask Questions**: Input questions about your data and receive answers using AI.

## Troubleshooting

- **Missing Dependencies**: Ensure all required packages are installed. Use `pip install -r requirements.txt` to install all dependencies.
- **API Initialization Errors**: Check your API key and configuration. Make sure `secrets.toml` is correctly set up.

## Contributing

Contributions are welcome! Please submit a pull request with a clear description of the changes and any relevant issues.
