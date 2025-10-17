import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Configuration and Model Loading ---

# Define the filename used when saving the model
MODEL_FILE = 'model/model.pkl'

# Define the target names for readable output
SPECIES_MAP = {
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
}

@st.cache_resource 
def load_model():
    """Loads the saved KNN model from the .pkl file."""
    try:
        # Load the model using joblib
        model = joblib.load(MODEL_FILE)
        return model
    except FileNotFoundError:
        # This error should now only occur if the file isn't on GitHub.
        st.error(f"Error: Model file '{MODEL_FILE}' not found on server.")
        return None

# Load the model once
knn_model = load_model()

# --- Streamlit UI Components ---

st.set_page_config(page_title="Iris Species Classifier", layout="centered")
st.title('Iris Flower Species Prediction')
st.markdown("---")

if knn_model is not None:
    st.markdown("### Enter the Flower Measurements (in cm):")
    
    # Create input widgets for the four features
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.slider('Sepal Length', min_value=4.0, max_value=8.0, value=5.5, step=0.1)
        petal_length = st.slider('Petal Length', min_value=1.0, max_value=7.0, value=3.5, step=0.1)
        
    with col2:
        sepal_width = st.slider('Sepal Width', min_value=2.0, max_value=4.5, value=3.0, step=0.1)
        petal_width = st.slider('Petal Width', min_value=0.1, max_value=2.5, value=1.2, step=0.1)
        
    st.markdown("---")
    
    # --- Prediction Logic ---
    if st.button('Predict Species', type="primary"):
        
        # 1. Prepare the input data as a NumPy array (in the format [SL, SW, PL, PW])
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # 2. Make the prediction (returns the species ID: 0, 1, or 2)
        prediction_id = knn_model.predict(input_data)[0]
        
        # 3. Map the ID to the Species Name
        predicted_species_name = SPECIES_MAP.get(prediction_id, "Unknown")
        
        # 4. Display the result
        st.success('Prediction Successful!')
        st.metric(
            label="Predicted Flower Species",
            value=f"{predicted_species_name}"
        )
        st.balloons()
else:
    st.warning("Cannot run prediction. Please ensure the model file is present.")