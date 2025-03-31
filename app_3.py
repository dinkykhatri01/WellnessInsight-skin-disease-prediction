import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
import os
import speech_recognition as sr
from googletrans import Translator, LANGUAGES
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from streamlit_cropper import st_cropper
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import custom_object_scope

# Define custom Cast layer with proper dtype handling
class Cast(Layer):
    def __init__(self, dtype, **kwargs):
        # Fix: Changed init to __init__
        self.dtype_value = dtype
        super(Cast, self).__init__(**kwargs)
        
    def call(self, inputs):
        return tf.cast(inputs, self.dtype_value)
        
    def get_config(self):
        config = super(Cast, self).get_config()
        config.update({"dtype": self.dtype_value})
        return config
    
    @classmethod
    def from_config(cls, config):
        dtype = config.pop("dtype")
        return cls(dtype=dtype, **config)

# Register the custom object so TensorFlow knows how to deserialize it
tf.keras.utils.get_custom_objects().update({'Cast': Cast})
# --------------------- Load Models and Data ---------------------
# Load CNN model for image-based disease classification
cnn_model = load_model('updated_vgg16_skin_disease.h5')

# Load segmentation model for preprocessing with custom layer
# First register the custom layer
tf.keras.utils.get_custom_objects().update({'Cast': Cast})

# Then load the model using the standard method
segmentation_model = load_model('RGB_segmentation_model_best.h5', compile=False)

# Optionally compile the model after loading if needed
segmentation_model.compile(optimizer='adam', loss='binary_crossentropy')

# Load symptom-based text classification model
with open('text_symptom_model.pkl', 'rb') as file:
    text_model = pickle.load(file)

# Load TF-IDF Vectorizer for text input
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Initialize translator
translator = Translator()

# --------------------- Streamlit Page Config ---------------------
st.set_page_config(
    page_title="Skin Disease Detector",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Add custom styles with gradient background and improved text visibility
st.markdown(
    """
    <style>
    /* Main app background - keeping the blue gradient */
    .stApp {
        background: linear-gradient(135deg, #E0F7FA, #4FC3F7, #0288D1);
        background-attachment: fixed;
    }
    
    /* Main title styling - added more padding to fix header overlap */
    .title {
        font-size: 52px;
        font-weight: 800;
        color: #003366;
        text-align: center;
        margin-bottom: 30px;
        margin-top: 40px; /* Added more top margin to prevent header overlap */
        padding-top: 30px; /* Added padding to help with header overlap */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 32px;
        font-weight: 700;
        color: #003366;
        text-align: center;
        margin: 40px 0 20px 0;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    /* Footer styling */
    .footer {
        font-size: 14px;
        color: #003366;
        text-align: center;
        margin-top: 40px;
        padding: 10px;
    }
    
    /* Button styling - changed to white */
    .stButton>button {
        background-color: white;
        color: #003366;
        border-radius: 5px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 600;
        border: 1px solid #003366;
        cursor: pointer;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #f8f8f8;
        box-shadow: 0 5px 10px rgba(0, 0, 0, 0.3);
        transform: translateY(-2px);
    }
    
    /* Text area styling - making sure background is white and text is black */
    .stTextArea>textarea {
        border-radius: 5px;
        padding: 12px;
        font-size: 16px;
        background-color: white !important;
        color: #000000 !important;
        border: 1px solid #003366;
    }
    
    /* File uploader styling - making the container white with black text */
    .stFileUploader {
        background-color: white !important;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
    }
    
    /* File uploader text color */
    .stFileUploader > div,
    .stFileUploader > div > div,
    .stFileUploader > div > div > div > div {
        color: black !important;
    }
    
    /* Camera input styling - white background with black text */
    .stCameraInput {
        background-color: white !important;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
    }
    
    .stCameraInput > div {
        background-color: white !important;
        border-radius: 10px;
        padding: 10px;
    }
    
    .stCameraInput > div div {
        color: black !important;
    }
    
    /* Upload and camera container styling */
    .upload-container {
        background-color: white !important;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Alert boxes */
    .stSuccess {
        background-color: rgba(76, 175, 80, 0.9);
        color: white;
        padding: 15px;
        border-radius: 5px;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    
    .stInfo {
        background-color: rgba(33, 150, 243, 0.9);
        color: white;
        padding: 15px;
        border-radius: 5px;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    
    .stWarning {
        background-color: rgba(255, 152, 0, 0.9);
        color: white;
        padding: 15px;
        border-radius: 5px;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    
    .stError {
        background-color: rgba(244, 67, 54, 0.9);
        color: white;
        padding: 15px;
        border-radius: 5px;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    
    /* Severity indicators */
    .mild-severity {
        background-color: rgba(76, 175, 80, 0.9);
        color: white;
        padding: 15px;
        border-radius: 5px;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        text-align: center;
        margin: 10px 0;
    }
    
    .moderate-severity {
        background-color: rgba(255, 152, 0, 0.9);
        color: white;
        padding: 15px;
        border-radius: 5px;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        text-align: center;
        margin: 10px 0;
    }
    
    .severe-severity {
        background-color: rgba(244, 67, 54, 0.9);
        color: white;
        padding: 15px;
        border-radius: 5px;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        text-align: center;
        margin: 10px 0;
    }
    
    /* Text colors for normal text */
    p, div:not(.stButton) {
        color: #000000;
    }
    
    /* Fixing header deploy button visibility issue */
    header button {
        background-color: white !important;
        color: #003366 !important;
        visibility: visible !important;
        opacity: 1 !important;
        border: 1px solid #003366 !important;
    }
    
    /* Remove the white containers */
    .block-container {
        padding-top: 3rem; /* Increased padding to fix header overlap */
        padding-bottom: 1rem;
    }
    
    /* OR divider */
    .divider {
        text-align: center;
        font-size: 20px;
        font-weight: 700;
        color: #003366;
        margin: 20px 0;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5);
    }
    
    /* Image display styling */
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Descriptive text styling */
    .description-text {
        background-color: rgba(0, 51, 102, 0.8);
        color: white !important;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        text-align: center;
        font-weight: 500;
    }
    
    /* Labels for uploads and text areas - now black */
    .stFileUploader label, .stTextArea label, .stCameraInput label {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 18px !important;
        margin-bottom: 10px !important;
    }
    
    /* Browse files button in file uploader - white */
    .stFileUploader div[data-testid="stFileUploadDropzone"] button {
        background-color: white !important;
        color: #003366 !important;
        border: 1px solid #003366 !important;
    }
    
    /* Make sure drop zone has white background */
    .stFileUploader div[data-testid="stFileUploadDropzone"] {
        background-color: white !important;
        color: black !important;
        border: 2px dashed #003366 !important;
    }
    
    /* Fix text input container */
    .stTextInput > div {
        background-color: white !important;
        color: black !important;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTextInput input {
        color: black !important;
    }
    
    /* Text area styling - making sure background is white and text is black */
    .stTextArea>div>div {
        background-color: white !important;
        border-radius: 10px;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
    }
    
    .stTextArea>textarea, .stTextArea div textarea {
        border-radius: 5px;
        padding: 12px;
        font-size: 16px;
        background-color: white !important;
        color: #000000 !important;
        border: 1px solid #003366;
    }
    
    /* Camera input button styling - making it white */
    .stCameraInput > div > div > button {
        background-color: white !important;
        color: #003366 !important;
        border: 1px solid #003366 !important;
        border-radius: 5px;
        padding: 8px 16px;
        font-weight: 600;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
    }
    
    /* Make sure the camera input container is white */
    .stCameraInput > div {
        background-color: white !important;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
    }
    
    /* Camera video container (when active) */
    .stCameraInput > div > div:first-child {
        background-color: white !important;
        border-radius: 5px;
    }
    
    /* File uploader (drag and drop box) styling */
    .stFileUploader > div {
        background-color: white !important;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
    }
    
    /* File uploader drop zone */
    .stFileUploader [data-testid="stFileUploadDropzone"] {
        background-color: white !important;
        border: 2px dashed #003366 !important;
        color: black !important;
    }
    
    /* All text inside file uploader */
    .stFileUploader [data-testid="stFileUploadDropzone"] p,
    .stFileUploader [data-testid="stFileUploadDropzone"] span,
    .stFileUploader [data-testid="stFileUploadDropzone"] div {
        color: black !important;
    }
    
    /* Camera input container (take photo box) */
    .stCameraInput {
        background-color: white !important;
        border-radius: 10px;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
        padding: 15px;
        margin-bottom: 20px;
    }
    
    /* All elements inside camera input */
    .stCameraInput div, 
    .stCameraInput p,
    .stCameraInput span {
        color: black !important;
    }
    
    /* Camera button */
    .stCameraInput button {
        background-color: white !important;
        color: #003366 !important;
        border: 1px solid #003366 !important;
        border-radius: 5px;
        font-weight: 600;
    }
    
    /* Text area (symptoms box) styling */
    .stTextArea {
        background-color: white !important;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    
    /* Text area input field */
    .stTextArea textarea {
        background-color: white !important;
        color: black !important;
        border: 1px solid #003366 !important;
        border-radius: 5px;
    }
    
    /* Container for all input elements */
    .css-ocqkz7, .css-1d391kg {
        background-color: white !important;
        border-radius: 10px;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
        padding: 15px;
        margin-bottom: 20px;
    }
    
    /* Make sure ALL text in input containers is black */
    .css-ocqkz7 *, .css-1d391kg * {
        color: black !important;
    }
    
    /* For specific classes that Streamlit might generate */
    div[data-baseweb="textarea"] {
        background-color: white !important;
    }
    
    div[data-baseweb="textarea"] textarea {
        color: black !important;
    }
    
    /* Force all labels to be black */
    label {
        color: black !important;
        font-weight: 600 !important;
    }
    
    /* Cropper interface styling */
    .cropper-container {
        background-color: white !important;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    
    .cropper-line, .cropper-point {
        background-color: #003366 !important; 
    }
    
    /* Crop button styling */
    .crop-button {
        background-color: white;
        color: #003366;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: 600;
        border: 1px solid #003366;
        margin-top: 10px;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .crop-button:hover {
        background-color: #f8f8f8;
        box-shadow: 0 5px 10px rgba(0, 0, 0, 0.3);
        transform: translateY(-2px);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------- Page Layout ---------------------
st.markdown('<p class="title">Skin Disease Detector 🩺</p>', unsafe_allow_html=True)
st.markdown(
    '<div class="description-text">'
    'Upload an image for skin disease prediction or enter your symptoms for analysis.'
    '</div>',
    unsafe_allow_html=True
)

# --------------------- Initialization for session state ---------------------
# Initialize session state for capturing live photo state
if "live_photo_captured" not in st.session_state:
    st.session_state["live_photo_captured"] = False
    
if "cropped_image" not in st.session_state:
    st.session_state["cropped_image"] = None
    
if "original_live_image" not in st.session_state:
    st.session_state["original_live_image"] = None
    
if "crop_completed" not in st.session_state:
    st.session_state["crop_completed"] = False
    
if "symptoms" not in st.session_state:
    st.session_state["symptoms"] = ""

def preprocess_image(image):
    img = image.resize((224,224))  # Resize image to model input
    img = np.array(img) / 255.0     # Normalize pixel values
    img = np.expand_dims(img, axis=0)
    return img

# Function to determine severity based on infection percentage
def determine_severity(infection_percentage):
    # Convert from string format (e.g., "85.25%") to float (85.25)
    if isinstance(infection_percentage, str):
        percentage = float(infection_percentage.strip('%'))
    else:
        percentage = infection_percentage
        
    # Define thresholds for severity levels
    if percentage < 40:
        return "Mild", "mild-severity"
    elif percentage < 70:
        return "Moderate", "moderate-severity"
    else:
        return "Severe", "severe-severity"

# --------------------- Image-Based Disease Prediction ---------------------
st.markdown('<p class="subtitle">Image-Based Prediction</p>', unsafe_allow_html=True)

# Option to upload image
uploaded_image = st.file_uploader("Upload an image of the affected skin (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

# Option to capture live image using camera
st.markdown('<div class="divider">OR</div>', unsafe_allow_html=True)

# Reset crop state if a new image is captured
live_camera = st.camera_input("Capture image using your camera")
if live_camera and not st.session_state["live_photo_captured"]:
    st.session_state["original_live_image"] = Image.open(live_camera)
    st.session_state["live_photo_captured"] = True
    st.session_state["crop_completed"] = False
    st.rerun()  # Using st.rerun() instead of experimental_rerun

# If a live photo was captured, offer cropping options
if st.session_state["live_photo_captured"] and not st.session_state["crop_completed"]:
    st.markdown('<div class="description-text">Crop the image to focus on the affected area</div>', unsafe_allow_html=True)
    
    # Use the st_cropper to crop the image
    cropped_img = st_cropper(
        st.session_state["original_live_image"],
        realtime_update=True,
        box_color='#003366',
        aspect_ratio=None
    )
    
    # Add a button to confirm cropping
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Confirm Crop"):
            st.session_state["cropped_image"] = cropped_img
            st.session_state["crop_completed"] = True
            st.rerun()  # Using st.rerun() instead of experimental_rerun

# Check if user uploaded an image
if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True, output_format="auto")
    
    # Predict using CNN model
    st.markdown('<div class="description-text">Predicting disease...</div>', unsafe_allow_html=True)
    processed_img = preprocess_image(image)
    predictions = cnn_model.predict(processed_img)

    # Display result
    class_names = ['Eczema','Melanoma','Normal','Psoriasis']
    predicted_class = class_names[np.argmax(predictions)]
    infection_percentage = np.max(predictions) * 100
    
    # Show prediction result
    st.success(f"Prediction: {predicted_class}")
    
    # Only show infection percentage and severity if NOT healthy
    if predicted_class != "Normal":
        infection_percentage_str = f"{infection_percentage:.2f}%"
        st.info(f"Infection Percentage: {infection_percentage_str}")
        
        # Determine and display severity
        severity_text, severity_class = determine_severity(infection_percentage)
        st.markdown(f'<div class="{severity_class}">Severity: {severity_text}</div>', unsafe_allow_html=True)

# Process the live captured image after cropping is confirmed
elif st.session_state["crop_completed"] and st.session_state["cropped_image"] is not None:
    # Display the cropped image
    st.image(st.session_state["cropped_image"], caption="Cropped Image", use_container_width=True, output_format="auto")
    
    # Predict using CNN model on the cropped image
    st.markdown('<div class="description-text">Predicting disease...</div>', unsafe_allow_html=True)
    processed_img = preprocess_image(st.session_state["cropped_image"])
    predictions = cnn_model.predict(processed_img)

    # Display result
    class_names = ['Eczema','Melanoma','Normal','Psoriasis']
    predicted_class = class_names[np.argmax(predictions)]
    infection_percentage = np.max(predictions) * 100
    
    # Show prediction result
    st.success(f"Prediction: {predicted_class}")
    
    # Only show infection percentage and severity if NOT healthy
    if predicted_class != "Normal":
        infection_percentage_str = f"{infection_percentage:.2f}%"
        st.info(f"Infection Percentage: {infection_percentage_str}")
        
        # Determine and display severity
        severity_text, severity_class = determine_severity(infection_percentage)
        st.markdown(f'<div class="{severity_class}">Severity: {severity_text}</div>', unsafe_allow_html=True)
    
    # Add option to retake photo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Retake Photo"):
            st.session_state["live_photo_captured"] = False
            st.session_state["cropped_image"] = None
            st.session_state["original_live_image"] = None
            st.session_state["crop_completed"] = False
            st.rerun()  # Using st.rerun() instead of experimental_rerun

# --------------------- Text-Based Symptom Prediction ---------------------
st.markdown('<p class="subtitle">Symptom-Based Prediction</p>', unsafe_allow_html=True)

# Text area for symptom input
st.session_state["symptoms"] = st.text_area(
    "Enter your symptoms (e.g., red patches, itchy skin, blisters):",
    value=st.session_state["symptoms"]
)

# Voice input button
st.markdown('<div class="divider">OR</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Speak Symptoms (English)"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening... Please speak your symptoms.")
            try:
                audio = r.listen(source, timeout=5)
                spoken_text = r.recognize_google(audio)  # Recognize in default language
                
                # Check if the text is in Hindi or English
                detected_lang = translator.detect(spoken_text).lang
                if detected_lang == "hi":  # If detected as Hindi, translate to English
                    st.session_state["symptoms"] = translator.translate(spoken_text, src="hi", dest="en").text
                else:  # If already English, use directly
                    st.session_state["symptoms"] = spoken_text
                    
            except sr.UnknownValueError:
                st.error("Sorry, I could not understand your speech. Please try again.")
            except sr.RequestError as e:
                st.error(f"Speech Recognition service is unavailable. {e}")

# Predict button for text-based symptoms
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Predict Disease", key="predict_text"):
        if st.session_state["symptoms"]:
            # Process symptom input
            text_features = tfidf_vectorizer.transform([st.session_state["symptoms"]])
            text_prediction = text_model.predict(text_features)[0]

            st.success(f"Predicted Disease: {text_prediction}")
        else:
            st.warning("Please enter symptoms or use the voice input to predict the disease.")

# Add requirements for the new package in a footer note
st.markdown(
    """
    <div class="footer">
    Note: This application requires the streamlit-cropper package. Install it with: pip install streamlit-cropper
    </div>
    """, 
    unsafe_allow_html=True
)