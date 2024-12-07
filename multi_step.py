import streamlit as st
from pathlib import Path
import tensorflow as tf
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import torch

# Define class names corresponding to model output
class_names = [
    'Green Light',
    'Red Light',
    'Speed Limit 10',
    'Speed Limit 100',
    'Speed Limit 110',
    'Speed Limit 120',
    'Speed Limit 20',
    'Speed Limit 30',
    'Speed Limit 40',
    'Speed Limit 50',
    'Speed Limit 60',
    'Speed Limit 70',
    'Speed Limit 80',
    'Speed Limit 90',
    'Stop'
]

# Page Configuration
st.set_page_config(
    page_title="Traffic Signs Detection App",
    page_icon="🚦",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Load custom CSS
def load_css(css_file_path):
    with open(css_file_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the CSS file
css_file_path = Path("styles.css")
if css_file_path.exists():
    load_css(css_file_path)
else:
    st.warning("CSS file not found. Please ensure 'styles.css' is in the same directory.")

# Cache the model loading to improve performance
@st.cache_resource
def load_classification_model():
    model = tf.keras.models.load_model('model.h5')  # Adjust path if necessary
    return model

# Cache YOLO model loading
@st.cache_resource
def load_yolo_model():
    # Make sure you have trained or downloaded a YOLO model for traffic sign detection
    # Replace 'path/to/your/traffic_sign_detection.pt' with your actual YOLO model path
    return YOLO('path/to/your/traffic_sign_detection.pt')

# Load models
classification_model = load_classification_model()
yolo_model = load_yolo_model()

# Header Section
st.markdown('<div class="main-header">Traffic Signs Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Easily upload and identify traffic signs!</div>', unsafe_allow_html=True)

# Upload Section
st.markdown("---")
st.header("🚀 Upload a Traffic Sign Image")
st.write("Upload an image of a traffic sign, and the model will detect and classify it for you.")

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Read the uploaded image
    img = Image.open(uploaded_file)
    
    # Convert PIL Image to OpenCV format (numpy array)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Perform object detection with YOLO
    results = yolo_model(img_cv)
    
    # Display the original image with detections
    annotated_img = results[0].plot()
    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), 
             caption="Detected Traffic Signs", 
             use_container_width=True)
    
    # Process each detected sign
    predictions = []
    for result in results[0]:
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, result.boxes.xyxy[0])
        
        # Crop the detected sign
        sign_crop = img_cv[y1:y2, x1:x2]
        
        # Convert cropped sign to PIL Image
        sign_pil = Image.fromarray(cv2.cvtColor(sign_crop, cv2.COLOR_BGR2RGB))
        
        # Preprocess the image for classification
        sign_pil = sign_pil.convert("L")  # Convert to grayscale (1 channel)
        sign_pil = sign_pil.resize((90, 90))  # Resize to match the model input size
        sign_array = np.array(sign_pil)
        
        # Normalize image values to be between 0 and 1
        sign_array = sign_array / 255.0
        
        # Reshape for CNN (add channel dimension)
        sign_array = np.expand_dims(sign_array, axis=-1)
        sign_array = np.expand_dims(sign_array, axis=0)
        
        # Predict using the trained classification model
        prediction = classification_model.predict(sign_array)
        predicted_class = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class]
        
        # Store prediction
        predictions.append({
            'bbox': (x1, y1, x2, y2),
            'class': predicted_class_name,
            'confidence': float(result.boxes.conf[0])
        })
    
    # Display predictions
    st.subheader("Detected Traffic Signs")
    for pred in predictions:
        st.write(f"Sign: {pred['class']} (Confidence: {pred['confidence']:.2f})")
        st.write(f"Bounding Box: {pred['bbox']}")

else:
    st.info("Please upload an image to get started.")

# Footer
st.markdown("---")
st.markdown('<div class="footer">Made with ❤️ using Streamlit and YOLO</div>', unsafe_allow_html=True)