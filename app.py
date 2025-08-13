import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Set the title and a brief description of the app
st.title('Cholera Case Prediction App')
st.markdown('This app uses your LSTM model to predict future cholera cases based on environmental and demographic data.')

# Use Streamlit's caching to load the model just once
@st.cache_resource
def load_my_model():
    """Load the pre-trained Keras model from the h5 file."""
    try:
        model = load_model('lstm_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_my_model()

if model:
    # --- User Input Section ---
    st.header("Enter Historical Data for Prediction")
    st.markdown("Please input a sequence of values for the past few weeks. This data will be used to make a future prediction.")

    # Define the features that the model was likely trained on
    features = [
        'Rainfall_mm', 'Avg_Temperature_C', 'Humidity_%', 
        'Water_Contamination_Index', 'Population_Density', 
        'Sanitation_Score', 'Is_Holiday_Week'
    ]

    # TODO: You must define the number of time steps (weeks) your model was trained on.
    # A common value is 5, but please check your model's architecture.
    TIMESTEPS = 5

    # Create a list to hold the input data
    input_data_list = []

    for i in range(TIMESTEPS):
        st.subheader(f'Data for Week {i+1}')
        week_data = {}
        for feature in features:
            if feature in ['Population_Density', 'Is_Holiday_Week']:
                # Input for integer-based features
                week_data[feature] = st.number_input(
                    f'{feature} (Week {i+1})', 
                    min_value=0, 
                    value=0, 
                    step=1, 
                    key=f'{feature}_{i}'
                )
            else:
                # Input for float-based features
                week_data[feature] = st.number_input(
                    f'{feature} (Week {i+1})', 
                    min_value=0.0, 
                    value=0.0, 
                    step=0.01, 
                    key=f'{feature}_{i}'
                )
        input_data_list.append(week_data)

    # --- Prediction Button and Logic ---
    if st.button('Predict Future Cholera Cases'):
        if input_data_list:
            # Convert the list of dictionaries to a DataFrame
            df_input = pd.DataFrame(input_data_list)
            
            # Convert the DataFrame to a numpy array
            input_array = df_input.values.astype(np.float32)

            # TODO: Reshape the data to match your LSTM's input shape.
            # Common shape is (batch_size, timesteps, features).
            # We assume batch_size=1, timesteps=TIMESTEPS, and features=len(features).
            input_reshaped = input_array.reshape(1, TIMESTEPS, len(features))

            # Make the prediction
            prediction = model.predict(input_reshaped)

            # --- Display the Result ---
            st.subheader("Prediction Result")
            st.success(f"Based on the provided data, the model predicts **{prediction[0][0]:.2f}** cholera cases for the next week.")
        else:
            st.error("Please enter data for all weeks to make a prediction.")

else:
    st.warning("Model failed to load. Please ensure 'lstm_model.h5' is in the same directory.")
