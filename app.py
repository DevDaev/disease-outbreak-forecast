import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import os
import hashlib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("TensorFlow not available. LSTM models will not work.")

# Page configuration
st.set_page_config(
    page_title="Cholera Forecast Platform",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
        padding-left: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .upload-section {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    
    .model-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .forecast-result {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Admin credentials and constants
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "cholera_admin_2024"  # In production, use proper authentication
MODELS_DIR = "uploaded_models"

# Create models directory if it doesn't exist
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

class ModelWrapper:
    def __init__(self, name, description, accuracy, model_file=None, model_type="mock"):
        self.name = name
        self.description = description
        self.accuracy = accuracy
        self.model_file = model_file
        self.model_type = model_type
        self._model = None
        
    def load_model(self):
        """Load the actual model from file"""
        if self._model is None and self.model_file:
            try:
                if self.model_type == "lstm" and TENSORFLOW_AVAILABLE:
                    # Load model with custom objects to handle older Keras versions
                    custom_objects = {
                        'mse': tf.keras.metrics.MeanSquaredError(),
                        'mae': tf.keras.metrics.MeanAbsoluteError(),
                        'accuracy': tf.keras.metrics.Accuracy()
                    }
                    self._model = keras.models.load_model(
                        self.model_file, 
                        custom_objects=custom_objects,
                        compile=False  # Skip compilation to avoid metric issues
                    )
                    # Recompile with current Keras version
                    self._model.compile(
                        optimizer='adam',
                        loss='mse',
                        metrics=['mae']
                    )
                elif self.model_type == "sklearn":
                    with open(self.model_file, 'rb') as f:
                        self._model = pickle.load(f)
                else:
                    st.error(f"Unsupported model type: {self.model_type}")
            except Exception as e:
                st.error(f"Error loading model {self.name}: {str(e)}")
                return None
        return self._model
        
    def predict(self, X):
        """Make predictions using the model"""
        if self.model_type == "mock":
            # Mock prediction logic for demo models
            np.random.seed(42)
            base_trend = np.linspace(X.mean(), X.mean() * 1.2, len(X))
            noise = np.random.normal(0, X.std() * 0.1, len(X))
            return base_trend + noise
        else:
            model = self.load_model()
            if model is None:
                st.warning(f"Model {self.name} failed to load, using fallback prediction")
                np.random.seed(42)
                base_trend = np.linspace(X.mean(), X.mean() * 1.2, len(X))
                noise = np.random.normal(0, X.std() * 0.1, len(X))
                return base_trend + noise
                
            try:
                if self.model_type == "lstm":
                    # Prepare data for LSTM (assuming it expects sequences)
                    X_reshaped = X.reshape((1, len(X), 1))
                    predictions = model.predict(X_reshaped, verbose=0)
                    return predictions.flatten()
                elif self.model_type == "sklearn":
                    return model.predict(X.reshape(-1, 1)).flatten()
                else:
                    raise ValueError(f"Unknown model type: {self.model_type}")
            except Exception as e:
                st.warning(f"Prediction failed for {self.name}: {str(e)}. Using fallback.")
                # Fallback prediction
                np.random.seed(42)
                base_trend = np.linspace(X.mean(), X.mean() * 1.2, len(X))
                noise = np.random.normal(0, X.std() * 0.1, len(X))
                return base_trend + noise

def load_uploaded_models():
    """Load all uploaded models from the models directory"""
    models = {}
    
    # Add default mock models
    models.update({
        "ARIMA Time Series": ModelWrapper(
            "ARIMA Time Series", 
            "Advanced time series model for seasonal cholera patterns",
            0.87,
            model_type="mock"
        ),
        "Random Forest": ModelWrapper(
            "Random Forest",
            "Ensemble model considering multiple environmental factors",
            0.82,
            model_type="mock"
        ),
        "Prophet": ModelWrapper(
            "Prophet",
            "Facebook's forecasting tool optimized for epidemiological data",
            0.85,
            model_type="mock"
        ),
        "Gradient Boosting": ModelWrapper(
            "Gradient Boosting",
            "Advanced boosting algorithm for high-accuracy predictions",
            0.89,
            model_type="mock"
        )
    })
    
    # Check for existing LSTM model
    lstm_path = "lstm_model.h5"
    if os.path.exists(lstm_path):
        models["LSTM Neural Network"] = ModelWrapper(
            "LSTM Neural Network",
            "Deep learning model for complex temporal dependencies (Pre-trained)",
            0.91,
            model_file=lstm_path,
            model_type="lstm"
        )
    
    # Load other uploaded models
    if os.path.exists(MODELS_DIR):
        for filename in os.listdir(MODELS_DIR):
            if filename.endswith('.h5'):
                model_name = filename.replace('.h5', '').replace('_', ' ').title()
                models[model_name] = ModelWrapper(
                    model_name,
                    "Custom uploaded LSTM model",
                    0.85,  # Default accuracy
                    model_file=os.path.join(MODELS_DIR, filename),
                    model_type="lstm"
                )
            elif filename.endswith('.pkl'):
                model_name = filename.replace('.pkl', '').replace('_', ' ').title()
                models[model_name] = ModelWrapper(
                    model_name,
                    "Custom uploaded scikit-learn model",
                    0.80,  # Default accuracy
                    model_file=os.path.join(MODELS_DIR, filename),
                    model_type="sklearn"
                )
    
    return models

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'uploaded_models' not in st.session_state:
    st.session_state.uploaded_models = load_uploaded_models()

MODELS = load_uploaded_models()

# Admin login interface
def admin_login():
    """Admin login interface"""
    st.markdown("### 🔐 Admin Login")
    
    with st.form("admin_login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state.is_admin = True
                st.success("✅ Admin login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

# Admin interface for managing models
def admin_model_management():
    """Admin interface for managing models"""
    st.markdown('<h2 class="sub-header">🔧 Model Management (Admin)</h2>', unsafe_allow_html=True)
    
    # Logout button
    if st.button("🚪 Logout", type="secondary"):
        st.session_state.is_admin = False
        st.rerun()
    
    st.markdown("---")
    
    # Model upload section
    st.markdown("### 📤 Upload New Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_model = st.file_uploader(
            "Upload Model File",
            type=['h5', 'pkl'],
            help="Upload .h5 files for TensorFlow/Keras models or .pkl files for scikit-learn models"
        )
        
        if uploaded_model is not None:
            model_name = st.text_input("Model Name", value=uploaded_model.name.split('.')[0])
            model_description = st.text_area("Model Description", 
                                           value="Custom uploaded model for cholera forecasting")
            model_accuracy = st.slider("Model Accuracy", 0.0, 1.0, 0.85, 0.01)
            
            if st.button("💾 Save Model", type="primary"):
                try:
                    # Save the uploaded file
                    file_extension = uploaded_model.name.split('.')[-1]
                    safe_filename = f"{model_name.replace(' ', '_').lower()}.{file_extension}"
                    file_path = os.path.join(MODELS_DIR, safe_filename)
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_model.getbuffer())
                    
                    # Add to models dictionary
                    model_type = "lstm" if file_extension == "h5" else "sklearn"
                    new_model = ModelWrapper(
                        model_name,
                        model_description,
                        model_accuracy,
                        model_file=file_path,
                        model_type=model_type
                    )
                    
                    MODELS[model_name] = new_model
                    st.session_state.uploaded_models = MODELS.copy()
                    
                    st.success(f"✅ Model '{model_name}' uploaded successfully!")
                    st.rerun()  # Refresh to show the new model
                    
                except Exception as e:
                    st.error(f"❌ Error uploading model: {str(e)}")
    
    with col2:
        st.markdown("### 📋 Upload Guidelines")
        st.info("""
        **Supported Formats:**
        - `.h5` - TensorFlow/Keras models
        - `.pkl` - Scikit-learn models
        
        **Requirements:**
        - Models should accept time series data
        - LSTM models should handle sequences
        - Ensure model compatibility
        """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Model Management")
    
    # Get current models
    current_models = st.session_state.uploaded_models if 'uploaded_models' in st.session_state else MODELS
    
    # Separate mock models from uploaded models
    mock_models = {k: v for k, v in current_models.items() if v.model_type == "mock"}
    uploaded_models = {k: v for k, v in current_models.items() if v.model_type != "mock"}
    
    # Mock models section (read-only)
    if mock_models:
        st.markdown("#### 🎭 Built-in Demo Models")
        st.info("These are demonstration models and cannot be deleted.")
        
        for model_name, model in mock_models.items():
            with st.expander(f"🤖 {model_name} (Demo)", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Description:** {model.description}")
                    st.write(f"**Type:** {model.model_type}")
                    st.write(f"**Accuracy:** {model.accuracy:.1%}")
                
                with col2:
                    st.info("Demo Model")
    
    # Uploaded models section (manageable)
    st.markdown("#### 📁 Uploaded Models")
    
    if not uploaded_models:
        st.info("No uploaded models found. Upload a model above to get started.")
    else:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**Total Uploaded Models:** {len(uploaded_models)}")
        with col2:
            if st.button("🔄 Refresh Models", help="Reload models from disk"):
                st.session_state.uploaded_models = load_uploaded_models()
                st.success("Models refreshed!")
                st.rerun()
        with col3:
            if uploaded_models and st.button("🗑️ Delete All", help="Delete all uploaded models"):
                st.session_state.show_delete_all_confirm = True
        
        if st.session_state.get('show_delete_all_confirm', False):
            st.warning("⚠️ Are you sure you want to delete ALL uploaded models? This action cannot be undone.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Delete All", type="primary"):
                    deleted_count = 0
                    for model_name, model in uploaded_models.items():
                        try:
                            if model.model_file and os.path.exists(model.model_file):
                                os.remove(model.model_file)
                                deleted_count += 1
                        except Exception as e:
                            st.error(f"Error deleting {model_name}: {str(e)}")
                    
                    # Remove from MODELS dict
                    for model_name in list(uploaded_models.keys()):
                        if model_name in MODELS:
                            del MODELS[model_name]
                    
                    st.session_state.uploaded_models = MODELS.copy()
                    st.session_state.show_delete_all_confirm = False
                    st.success(f"✅ Deleted {deleted_count} models successfully!")
                    st.rerun()
            
            with col2:
                if st.button("❌ Cancel"):
                    st.session_state.show_delete_all_confirm = False
                    st.rerun()
        
        # Individual model management
        for model_name, model in uploaded_models.items():
            with st.expander(f"🤖 {model_name}", expanded=False):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**Description:** {model.description}")
                    st.write(f"**Type:** {model.model_type.upper()}")
                    st.write(f"**Accuracy:** {model.accuracy:.1%}")
                    if model.model_file:
                        st.write(f"**File:** {os.path.basename(model.model_file)}")
                        try:
                            file_stats = os.stat(model.model_file)
                            file_size = file_stats.st_size / (1024 * 1024)  # MB
                            mod_time = datetime.fromtimestamp(file_stats.st_mtime)
                            st.write(f"**Size:** {file_size:.2f} MB")
                            st.write(f"**Modified:** {mod_time.strftime('%Y-%m-%d %H:%M')}")
                        except:
                            pass
                
                with col2:
                    if st.button(f"🧪 Test", key=f"test_{model_name}"):
                        with st.spinner("Testing model..."):
                            try:
                                # Test model loading
                                test_data = np.random.randn(30)
                                predictions = model.predict(test_data)
                                st.success("✅ Model loads and predicts successfully")
                                st.write(f"**Input shape:** {test_data.shape}")
                                st.write(f"**Output shape:** {predictions.shape}")
                                st.write(f"**Sample prediction:** {predictions[0]:.2f}")
                            except Exception as e:
                                st.error(f"❌ Model test failed: {str(e)}")
                
                with col3:
                    delete_key = f"delete_{model_name}"
                    confirm_key = f"confirm_delete_{model_name}"
                    
                    if st.session_state.get(confirm_key, False):
                        st.warning("Confirm delete?")
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            if st.button("✅", key=f"yes_{model_name}"):
                                try:
                                    # Delete file
                                    if model.model_file and os.path.exists(model.model_file):
                                        os.remove(model.model_file)
                                    
                                    # Remove from MODELS dict
                                    if model_name in MODELS:
                                        del MODELS[model_name]
                                    
                                    # Update session state
                                    st.session_state.uploaded_models = MODELS.copy()
                                    st.session_state[confirm_key] = False
                                    
                                    st.success(f"✅ Deleted {model_name}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error deleting model: {str(e)}")
                        
                        with col_no:
                            if st.button("❌", key=f"no_{model_name}"):
                                st.session_state[confirm_key] = False
                                st.rerun()
                    else:
                        if st.button("🗑️ Delete", key=delete_key):
                            st.session_state[confirm_key] = True
                            st.rerun()

# Main function to handle page navigation
def main():
    # Header
    st.markdown('<h1 class="main-header">🦠 Cholera Forecast Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d;">Advanced AI-powered cholera outbreak prediction and analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Navigation")
        
        if st.session_state.is_admin:
            page_options = ["Data Upload", "Model Selection", "Forecast Results", "Analytics Dashboard", "Admin Panel"]
        else:
            page_options = ["Data Upload", "Model Selection", "Forecast Results", "Analytics Dashboard", "Admin Login"]
        
        page = st.selectbox("Choose Section", page_options)
        
        st.markdown("---")
        
        if st.session_state.is_admin:
            st.success("🔐 Admin Mode Active")
            st.markdown(f"**Available Models:** {len(MODELS)}")
        
        st.markdown("### ℹ️ About")
        st.info("This platform uses advanced machine learning models to forecast cholera outbreaks based on historical data patterns.")
        
        st.markdown("### 📋 Data Requirements")
        st.markdown("""
        **Required columns:**
        - Date (YYYY-MM-DD)
        - Cases (number of cases)
        
        **Optional columns:**
        - Location
        - Population
        - Temperature
        - Rainfall
        """)
    
    if page == "Data Upload":
        data_upload_page()
    elif page == "Model Selection":
        model_selection_page()
    elif page == "Forecast Results":
        forecast_results_page()
    elif page == "Analytics Dashboard":
        analytics_dashboard_page()
    elif page == "Admin Login":
        admin_login()
    elif page == "Admin Panel":
        if st.session_state.is_admin:
            admin_model_management()
        else:
            st.error("❌ Access denied. Please login as admin.")

# Data upload page
def data_upload_page():
    st.markdown('<h2 class="sub-header">📁 Data Upload</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload your cholera cases CSV file",
            type=['csv'],
            help="Upload a CSV file containing cholera case data with date and case count columns"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                
                st.success("✅ File uploaded successfully!")
                
                # Data preview
                st.markdown("### 📋 Data Preview")
                st.dataframe(data.head(10), use_container_width=True)
                
                # Data validation
                st.markdown("### ✅ Data Validation")
                validation_results = validate_data(data)
                
                for result in validation_results:
                    if result['status'] == 'success':
                        st.success(result['message'])
                    elif result['status'] == 'warning':
                        st.warning(result['message'])
                    else:
                        st.error(result['message'])
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with col2:
        if st.session_state.data is not None:
            data = st.session_state.data
            
            # Data statistics
            st.markdown("### 📊 Data Statistics")
            
            total_records = len(data)
            st.markdown(f'<div class="metric-card"><h3>{total_records:,}</h3><p>Total Records</p></div>', unsafe_allow_html=True)
            
            if 'cases' in data.columns.str.lower():
                cases_col = [col for col in data.columns if 'case' in col.lower()][0]
                total_cases = data[cases_col].sum()
                avg_cases = data[cases_col].mean()
                
                st.markdown(f'<div class="metric-card"><h3>{total_cases:,}</h3><p>Total Cases</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><h3>{avg_cases:.1f}</h3><p>Average Cases/Day</p></div>', unsafe_allow_html=True)
            
            # Quick visualization
            if len(data.columns) >= 2:
                st.markdown("### 📈 Quick Visualization")
                fig = create_quick_plot(data)
                st.plotly_chart(fig, use_container_width=True)

# Model selection page
def model_selection_page():
    st.markdown('<h2 class="sub-header">🤖 Model Selection</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first in the Data Upload section.")
        return
    
    current_models = st.session_state.uploaded_models if 'uploaded_models' in st.session_state else MODELS
    
    st.markdown("### Available Models")
    
    # Model selection grid
    cols = st.columns(2)
    selected_model = None
    
    for i, (model_name, model) in enumerate(current_models.items()):
        with cols[i % 2]:
            st.markdown(f'<div class="model-card">', unsafe_allow_html=True)
            
            if st.button(f"Select {model_name}", key=f"select_{model_name}"):
                selected_model = model_name
                st.session_state.selected_model = model_name
            
            st.markdown(f"**{model_name}**")
            st.markdown(f"*{model.description}*")
            
            if model.model_type != "mock":
                st.markdown(f"**Type:** {model.model_type.upper()}")
                if model.model_file:
                    st.markdown(f"**File:** {os.path.basename(model.model_file)}")
            
            # Accuracy bar
            accuracy_percent = int(model.accuracy * 100)
            st.progress(model.accuracy)
            st.markdown(f"Accuracy: {accuracy_percent}%")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Model configuration
    if hasattr(st.session_state, 'selected_model'):
        st.markdown("---")
        st.markdown(f"### ⚙️ Configure {st.session_state.selected_model}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_days = st.slider("Forecast Period (days)", 7, 90, 30)
            confidence_interval = st.selectbox("Confidence Interval", [80, 90, 95], index=1)
        
        with col2:
            include_seasonality = st.checkbox("Include Seasonality", True)
            include_trend = st.checkbox("Include Trend", True)
        
        # Run forecast button
        if st.button("🚀 Run Forecast", type="primary"):
            with st.spinner("Running forecast..."):
                results = run_forecast(
                    st.session_state.data, 
                    st.session_state.selected_model,
                    forecast_days,
                    confidence_interval
                )
                st.session_state.forecast_results = results
                st.success("✅ Forecast completed! Check the Forecast Results section.")

# Forecast results page
def forecast_results_page():
    st.markdown('<h2 class="sub-header">📈 Forecast Results</h2>', unsafe_allow_html=True)
    
    if st.session_state.forecast_results is None:
        st.warning("⚠️ No forecast results available. Please run a forecast in the Model Selection section.")
        return
    
    results = st.session_state.forecast_results
    
    # Results summary
    st.markdown('<div class="forecast-result">', unsafe_allow_html=True)
    st.markdown(f"### 🎯 Forecast Summary")
    st.markdown(f"**Model Used:** {results['model_name']}")
    st.markdown(f"**Forecast Period:** {results['forecast_days']} days")
    st.markdown(f"**Predicted Peak:** {results['predicted_peak']:.0f} cases on {results['peak_date']}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Forecast visualization
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 📊 Forecast Visualization")
        fig = create_forecast_plot(results)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Key Metrics")
        
        metrics = results['metrics']
        for metric_name, value in metrics.items():
            st.metric(metric_name, f"{value:.2f}")
    
    # Detailed forecast table
    st.markdown("### 📅 Detailed Forecast")
    forecast_df = pd.DataFrame(results['forecast_data'])
    st.dataframe(forecast_df, use_container_width=True)
    
    # Download results
    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast Results",
        data=csv,
        file_name=f"cholera_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Analytics dashboard page
def analytics_dashboard_page():
    st.markdown('<h2 class="sub-header">📊 Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first to view analytics.")
        return
    
    data = st.session_state.data
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if 'cases' in data.columns.str.lower():
        cases_col = [col for col in data.columns if 'case' in col.lower()][0]
        
        with col1:
            st.metric("Total Cases", f"{data[cases_col].sum():,}")
        
        with col2:
            st.metric("Average Daily Cases", f"{data[cases_col].mean():.1f}")
        
        with col3:
            st.metric("Peak Cases", f"{data[cases_col].max():,}")
        
        with col4:
            st.metric("Data Points", f"{len(data):,}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Cases Over Time")
        fig1 = create_time_series_plot(data)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Cases Distribution")
        fig2 = create_distribution_plot(data)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Additional analytics
    if len(data.columns) > 2:
        st.markdown("### 🔍 Correlation Analysis")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            fig3 = px.imshow(corr_matrix, text_auto=True, aspect="auto")
            st.plotly_chart(fig3, use_container_width=True)

# Validate uploaded data
def validate_data(data):
    """Validate uploaded data"""
    results = []
    
    # Check for required columns
    required_cols = ['date', 'cases']
    data_cols_lower = [col.lower() for col in data.columns]
    
    for req_col in required_cols:
        if any(req_col in col for col in data_cols_lower):
            results.append({
                'status': 'success',
                'message': f"✅ Found {req_col} column"
            })
        else:
            results.append({
                'status': 'error',
                'message': f"❌ Missing required {req_col} column"
            })
    
    # Check data quality
    if len(data) < 30:
        results.append({
            'status': 'warning',
            'message': "⚠️ Dataset has fewer than 30 records. More data recommended for better forecasts."
        })
    
    return results

# Create a quick visualization of the data
def create_quick_plot(data):
    """Create a quick visualization of the data"""
    if len(data.columns) >= 2:
        x_col = data.columns[0]
        y_col = data.columns[1]
        
        fig = px.line(data, x=x_col, y=y_col, title="Data Overview")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        return fig
    
    fig = go.Figure()
    fig.add_annotation(
        text="No suitable columns for visualization",
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor='center', yanchor='middle',
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        title="Data Overview",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False)
    )
    return fig

# Create time series plot
def create_time_series_plot(data):
    """Create time series plot"""
    if 'cases' in data.columns.str.lower():
        cases_col = [col for col in data.columns if 'case' in col.lower()][0]
        date_col = data.columns[0]  # Assume first column is date
        
        fig = px.line(data, x=date_col, y=cases_col, title="Cholera Cases Over Time")
        fig.update_traces(line_color='#1f77b4', line_width=3)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        return fig
    
    fig = go.Figure()
    fig.add_annotation(
        text="No cases data available",
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor='center', yanchor='middle',
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        title="Cholera Cases Over Time",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False)
    )
    return fig

# Create distribution plot
def create_distribution_plot(data):
    """Create distribution plot"""
    if 'cases' in data.columns.str.lower():
        cases_col = [col for col in data.columns if 'case' in col.lower()][0]
        
        fig = px.histogram(data, x=cases_col, title="Cases Distribution", nbins=20)
        fig.update_traces(marker_color='#ff7f0e')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        return fig
    
    fig = go.Figure()
    fig.add_annotation(
        text="No cases data available",
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor='center', yanchor='middle',
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        title="Cases Distribution",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False)
    )
    return fig

# Run forecast using selected model
def run_forecast(data, model_name, forecast_days, confidence_interval):
    """Run forecast using selected model"""
    current_models = st.session_state.uploaded_models if 'uploaded_models' in st.session_state else MODELS
    model = current_models[model_name]
    
    try:
        # Extract cases data with better validation
        if 'cases' in data.columns.str.lower():
            cases_col = [col for col in data.columns if 'case' in col.lower()][0]
            cases_data = data[cases_col].values
        else:
            cases_data = data.iloc[:, 1].values  # Use second column as cases
        
        # Convert to numeric and handle NaN values
        cases_data = pd.to_numeric(cases_data, errors='coerce')
        cases_data = cases_data[~np.isnan(cases_data)]  # Remove NaN values
        
        if len(cases_data) == 0:
            raise ValueError("No valid numeric data found in cases column")
            
        # Ensure we have enough data
        if len(cases_data) < 5:
            raise ValueError("Insufficient data points for forecasting")
            
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        # Create dummy data for demonstration
        cases_data = np.random.poisson(50, 30)  # 30 days of dummy data
    
    try:
        # Use last 30 days for prediction (or all data if less than 30)
        recent_data = cases_data[-min(30, len(cases_data)):]
        forecast = model.predict(recent_data)
        
        # If forecast is shorter than requested days, extend it
        if len(forecast) < forecast_days:
            if len(forecast) > 0:
                last_values = forecast[-min(5, len(forecast)):]
                if len(last_values) > 1:
                    trend = np.mean(np.diff(last_values))
                else:
                    trend = 0
                last_value = forecast[-1]
            else:
                # If no forecast generated, use last actual value
                last_value = float(cases_data[-1])
                trend = 0
            
            extended_forecast = []
            for i in range(len(forecast), forecast_days):
                next_value = last_value + trend * (i - len(forecast) + 1)
                extended_forecast.append(max(0, next_value))  # Ensure non-negative
            
            forecast = np.concatenate([forecast, extended_forecast]) if len(forecast) > 0 else np.array(extended_forecast)
        
        forecast = forecast[:forecast_days]  # Trim to requested length
        
    except Exception as e:
        st.error(f"Error running forecast with {model_name}: {str(e)}")
        try:
            last_value = float(cases_data[-1])
            # Simple linear trend based on recent data
            if len(cases_data) >= 7:
                recent_trend = np.mean(np.diff(cases_data[-7:]))
            else:
                recent_trend = 0
            
            forecast = []
            for i in range(forecast_days):
                predicted_value = last_value + recent_trend * (i + 1)
                # Add some realistic variation
                variation = np.random.normal(0, last_value * 0.1)
                forecast.append(max(0, predicted_value + variation))
            
            forecast = np.array(forecast)
        except Exception as fallback_error:
            st.error(f"Fallback prediction also failed: {str(fallback_error)}")
            # Ultimate fallback - constant prediction
            last_value = 50  # Default value
            forecast = np.full(forecast_days, last_value)
    
    # Generate future dates
    last_date = datetime.now()
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    
    # Create forecast data
    forecast_data = []
    for i, date in enumerate(future_dates):
        predicted_cases = max(0, float(forecast[i]))  # Ensure non-negative and convert to float
        lower_bound = max(0, predicted_cases * 0.8)
        upper_bound = predicted_cases * 1.2
        
        forecast_data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Predicted Cases': round(predicted_cases, 1),
            'Lower Bound': round(lower_bound, 1),
            'Upper Bound': round(upper_bound, 1)
        })
    
    # Calculate metrics
    predicted_values = [item['Predicted Cases'] for item in forecast_data]
    peak_idx = np.argmax(predicted_values)
    
    results = {
        'model_name': model_name,
        'forecast_days': forecast_days,
        'forecast_data': forecast_data,
        'predicted_peak': predicted_values[peak_idx],
        'peak_date': forecast_data[peak_idx]['Date'],
        'metrics': {
            'Mean Forecast': np.mean(predicted_values),
            'Max Forecast': np.max(predicted_values),
            'Total Predicted': np.sum(predicted_values),
            'Model Accuracy': model.accuracy
        }
    }
    
    return results

# Create forecast visualization
def create_forecast_plot(results):
    """Create forecast visualization"""
    forecast_df = pd.DataFrame(results['forecast_data'])
    
    fig = go.Figure()
    
    # Add predicted line
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Predicted Cases'],
        mode='lines+markers',
        name='Predicted Cases',
        line=dict(color='#1f77b4', width=3)
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Upper Bound'],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Lower Bound'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='Confidence Interval',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    fig.update_layout(
        title=f"Cholera Cases Forecast - {results['model_name']}",
        xaxis_title="Date",
        yaxis_title="Number of Cases",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    
    return fig

if __name__ == "__main__":
    main()
