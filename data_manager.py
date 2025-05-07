import streamlit as st
import pickle
import numpy as np
import os
import scipy.stats as stats

# Path to models folder
MODELS_PATH = "models/"

# List of available feature extraction methods
FEATURE_METHODS = [
    "Raw Pixel Values",
    "Matrix Properties (Pixel,Rank,Det,Trace)",
    "Matrix Properties (Rank,Det,Trace)",
    "Pixels and MPs of Scalograms(CWT,STFT)",
    "MPs of Original,CWT,STFT",
    "Pixels of Original and MPs of Original,CWT,STFT",
    "Pixels of Original,CWT,STFT and MPs of Original,CWT,STFT",
    "Smith Normal Form With Window Size 5"
]

# Mapping of feature extraction methods to their corresponding model filenames
MODEL_MAPPING = {
    method: f"{method.lower().replace(' ', '_').replace('(', '').replace(')', '')}_model.sav" 
    for method in FEATURE_METHODS
}

def initialize_session_state():
    """Initialize all session state variables if they don't exist"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'welcome'
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'selected_feature' not in st.session_state:
        st.session_state.selected_feature = None
    if 'processing_steps' not in st.session_state:
        st.session_state.processing_steps = []
    if 'processing_visualizations' not in st.session_state:
        st.session_state.processing_visualizations = []
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    if 'patient_info' not in st.session_state:
        st.session_state.patient_info = {"name": "", "age": 45, "gender": "Male"}
    if 'report' not in st.session_state:
        st.session_state.report = None

def load_model(feature_method,classification):
    """Load the appropriate model for the selected feature extraction method"""
    try:
        model_filename = MODEL_MAPPING[feature_method]
        model_path = os.path.join(MODELS_PATH, classification)
        model_path=os.path.join(model_path, model_filename)
        
        # Append status to processing steps (for Streamlit UI)
        st.session_state.processing_steps.append(f"Loading model for {feature_method}")

        # Load and return the actual model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def extract_features(image, feature_method):
    """Extract features from the image using the selected method"""
    from utils import preprocess_image
    from utils import feature_extraction
    from utils import feature_normalisation
    
    # Add processing step to log
    st.session_state.processing_steps.append(f"Applying {feature_method} to extract features")
    
    # Get preprocessing steps and visualizations
    processing_steps = preprocess_image(image, feature_method)
    
    # Simulate feature vector
    feature_vector = feature_extraction(image,feature_method)
    normalised_feature_vector=feature_normalisation(feature_vector,feature_method)


    
    return normalised_feature_vector, processing_steps

# def predict_disease(features, model):
#     """Make predictions using the model"""
#     # Add processing step to log
#     st.session_state.processing_steps.append("Generating predictions based on extracted features")
    
#     # Simulate prediction probabilities
#     probs = np.random.rand(len(DISEASE_CLASSES))
#     probs = probs / probs.sum()  # Normalize to sum to 1
    
#     # Sort predictions by probability
#     predictions = [(class_name, prob) for class_name, prob in zip(DISEASE_CLASSES, probs)]
#     predictions.sort(key=lambda x: x[1], reverse=True)
#     st.session_state.label=predictions[0]
    
#     return predictions

def predict_disease(features, model):
    """Make predictions using the trained model"""
    # Add processing step to log
    st.session_state.processing_steps.append("Generating predictions based on extracted features")

    # Convert to numpy and reshape
    features = np.array(features).astype(np.complex128).reshape(1, -1)
    
    # Remove complex type (only use real part or abs value)
    real_features = np.abs(features.real)  # or: features.real if imaginary part is always zero

    # Get predicted probabilities
    probs = model.predict_proba(real_features)[0]

    # Sort predictions by confidence
    predictions = [[class_name, float(prob)] for class_name, prob in zip(model.classes_, probs)]
    if st.session_state.selected_classification == "CN":
        predictions[0][0] = "Normal"
        predictions[1][0] = "COVID-19"
    elif st.session_state.selected_classification == "CP":
        predictions[0][0] = "COVID-19"
        predictions[1][0] = "Pneumonia"
    elif st.session_state.selected_classification == "NP":
        predictions[0][0] = "Normal"
        predictions[1][0] = "Pneumonia"
    elif st.session_state.selected_classification == "CNP":
        predictions[0][0] = "Normal"
        predictions[1][0] = "COVID-19"
        predictions[2][0] = "Pneumonia"
    predictions.sort(key=lambda x: x[1], reverse=True)
    # Save top prediction
    st.session_state.label = predictions[0]
    return predictions



def generate_report(patient_info, predictions, feature_method):
    """Generate a medical report with the prediction results"""
    top_disease, top_prob = predictions[0]
    second_disease, second_prob = predictions[1]
    
    report = {
        "patient_info": {
            "name": patient_info["name"],
            "age": patient_info["age"],
            "gender": patient_info["gender"],
            "exam_date": st.session_state.get("exam_date", "2025-04-16"),
        },
        "analysis": {
            "method": feature_method,
            "primary_finding": top_disease,
            "primary_confidence": f"{top_prob:.2%}",
            "differential_diagnosis": second_disease,
            "differential_confidence": f"{second_prob:.2%}",
            "recommendations": get_recommendations(top_disease),
        },
        "all_probabilities": {name: f"{prob:.2%}" for name, prob in predictions}
    }
    
    return report

def get_recommendations(disease):
    """Get recommendations based on predicted disease"""
    recommendations = {
        "Normal": "No specific follow-up required. Maintain regular check-ups.",
        "Pneumonia": "Recommend antibiotics and follow-up chest X-ray in 4-6 weeks.",
        "COVID-19": "Isolate patient, perform PCR test, monitor oxygen levels.",
    }
    
    return recommendations.get(disease, "Follow up with specialist for further evaluation.")

def reset_analysis():
    """Reset analysis-related session state variables"""
    st.session_state.uploaded_image = None
    st.session_state.selected_feature = None
    st.session_state.processing_steps = []
    st.session_state.processing_visualizations = []
    st.session_state.prediction_results = None
    st.session_state.report = None