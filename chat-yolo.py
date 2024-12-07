import streamlit as st
from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import torch  # PyTorch for YOLO inference

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


# @st.cache_resource
# def load_yolo_model():
#     # Load the YOLOv8 model using ultralytics
#     model = YOLO("fine_tuned_yolov8s.pt")  # Replace with the path to your fine-tuned YOLOv8 model
#     return model
# Load the YOLO model
@st.cache_resource
def load_yolo_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

yolo_model = load_yolo_model()

# Load the trained classifier model
@st.cache_resource
def load_classifier_model():
    model = tf.keras.models.load_model('model.h5')  # Adjust path if necessary
    return model

classifier_model = load_classifier_model()

# Header Section
st.markdown('<div class="main-header">Traffic Signs Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Easily upload and identify traffic signs!</div>', unsafe_allow_html=True)

# Upload Section
st.markdown("---")
st.header("🚀 Upload a Traffic Sign Image")
st.write("Upload an image of a traffic sign, and the model will detect and classify it.")

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Traffic Sign", use_container_width=True)
    st.success("Image uploaded successfully!")

    # Load the image
    img = Image.open(uploaded_file)
    img_rgb = img.convert("RGB")  # Ensure the image is in RGB format

    # Perform object detection with YOLO
    st.write("Detecting traffic signs...")
    results = yolo_model(img_rgb)
    detections = results.xyxy[0].numpy()  # Extract bounding box and confidence scores

# Display detected objects with their respective class names and confidence scores
# Display detected objects with class names and confidence scores in a grid layout
    if len(detections) > 0:
        st.markdown("### Detected Traffic Signs")

        # Draw bounding boxes on the original image
        draw = ImageDraw.Draw(img)
        for det in detections:
            x1, y1, x2, y2, _, _ = det
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Display the image with all bounding boxes
        st.image(img, caption="Detected Traffic Signs with Bounding Boxes", use_container_width=True)

        # Create a grid layout for individual detections
        num_cols = 3  # Define the number of columns in the grid
        cols = st.columns(num_cols)

        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = map(int, det)
            cropped_img = img_rgb.crop((x1, y1, x2, y2))

            # Preprocess the cropped image for classification
            gray_img = cropped_img.convert("L").resize((90, 90))
            img_array = np.array(gray_img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Classify using the CNN model
            prediction = classifier_model.predict(img_array)
            predicted_class_idx = np.argmax(prediction)
            predicted_class_name = class_names[predicted_class_idx]
            confidence_score = np.max(prediction) * 100  # Convert to percentage

            # Display the result in a card-like style
            col = cols[i % num_cols]  # Cycle through columns
            with col:
                st.image(cropped_img, caption=f"Sign {i + 1}", use_column_width=True)
                st.markdown(
                    f"""
                    <div style="text-align: center; margin-top: -10px;">
                        <strong>{predicted_class_name}</strong><br>
                        <span style="color: gray;">Confidence: {confidence_score:.2f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.warning("No traffic signs detected in the image.")


else:
    st.info("Please upload an image to get started.")

# Footer
st.markdown("---")
st.markdown('<div class="footer">Made with ❤️ using Streamlit</div>', unsafe_allow_html=True)
