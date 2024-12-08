import streamlit as st
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

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

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.h5')  # Adjust path if necessary
    return model

model = load_model()

# Header Section
st.markdown('<div class="main-header">Traffic Signs Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Easily upload and identify traffic signs!</div>', unsafe_allow_html=True)

# Upload Section
st.markdown("---")
st.header("🚀 Upload a Traffic Sign Image")
st.write("Upload an image of a traffic sign, and the model will classify it for you.")

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Traffic Sign", use_container_width=True)
    st.success("Image uploaded successfully!")

    # Preprocess the image for prediction
    img = Image.open(uploaded_file)
    img = img.resize((90, 90))  # Resize to match the model input size
    img = np.array(img)  # Convert to numpy array

    # Normalize image values to be between 0 and 1 (if required by your model)
    img = img / 255.0  # Adjust based on your model's preprocessing

    # Ensure the image has 3 channels (RGB)
    if img.shape[-1] != 3:
        st.warning("The image does not have 3 channels (RGB). Please upload a valid colored image.")
    else:
        # Add batch dimension (model expects input of shape (batch_size, height, width, channels))
        img = np.expand_dims(img, axis=0)  # Now shape will be (1, 90, 90, 3)

        # Predict using the trained model
        prediction = model.predict(img)

        # Get the predicted class index
        predicted_class = np.argmax(prediction)  # Get the class with the highest probability

        # Map the predicted class to its corresponding label
        predicted_class_name = class_names[predicted_class]

        # Display the prediction (class name)
        st.write(f"Predicted Class: {predicted_class_name}")

else:
    st.info("Please upload an image to get started.")

# Footer
st.markdown("---")
st.markdown('<div class="footer">Made with ❤️ using Streamlit</div>', unsafe_allow_html=True)
