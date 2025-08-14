# Cholera Forecast Platform

A comprehensive Streamlit application for cholera outbreak prediction using machine learning models.

## Features

- **Interactive Data Upload**: Easy CSV file upload with data validation
- **Multiple ML Models**: Choose from ARIMA, Random Forest, LSTM, Prophet, and Gradient Boosting
- **Real-time Forecasting**: Generate predictions with confidence intervals
- **Analytics Dashboard**: Comprehensive data visualization and analysis
- **Modern UI**: Clean, responsive interface with interactive charts

## Installation

1. Install required packages:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

2. Run the application:
\`\`\`bash
streamlit run app.py
\`\`\`

## Usage

1. **Data Upload**: Upload your cholera cases CSV file with date and cases columns
2. **Model Selection**: Choose from available pre-trained models and configure parameters
3. **Run Forecast**: Generate predictions for your specified time period
4. **View Results**: Analyze forecast results and download predictions
5. **Analytics**: Explore your data with interactive visualizations

## Data Format

Your CSV file should contain:
- **Date column**: Date in YYYY-MM-DD format
- **Cases column**: Number of cholera cases
- **Optional**: Location, population, temperature, rainfall data

## Sample Data

Run `python sample_data_generator.py` to generate sample cholera data for testing.

## Models Available

- **ARIMA Time Series**: Advanced time series model for seasonal patterns
- **Random Forest**: Ensemble model considering environmental factors
- **LSTM Neural Network**: Deep learning for complex temporal dependencies
- **Prophet**: Facebook's forecasting tool for epidemiological data
- **Gradient Boosting**: Advanced boosting algorithm for high accuracy

## Features

- Data validation and quality checks
- Interactive visualizations with Plotly
- Confidence intervals for predictions
- Downloadable forecast results
- Responsive design for all devices
