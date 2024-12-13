import streamlit as st
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import base64
from io import BytesIO

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

st.set_page_config(
    page_title="Traffic Signs Detection App",
    page_icon="🚦",
    layout="centered",
    initial_sidebar_state="expanded",
)

def load_css(css_file_path):
    with open(css_file_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_file_path = Path("styles.css")
if css_file_path.exists():
    load_css(css_file_path)
else:
    st.warning("CSS file not found. Please ensure 'styles.css' is in the same directory.")

# Load the YOLO model for detection
@st.cache_resource
def load_yolo_model():
    return YOLO("models/best.onnx") 

yolo_model = load_yolo_model()

# Load the trained CNN model
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model('models/model.keras')

cnn_model = load_cnn_model()

def image_to_base64(img_array):
    buffered = BytesIO()
    img = Image.fromarray(img_array) 
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

st.markdown('<div class="main-header">Traffic Signs Detection App</div>', unsafe_allow_html=True)

st.markdown("---")
st.header("Upload a Traffic Sign Image")
st.write("Upload an image of a traffic sign, and the model will classify it for you.")

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)

    if img.mode != "RGB":
        img = img.convert("RGB")
    img_array = np.array(img)  

    results = yolo_model(img_array)
    detections = results[0].boxes.xyxy.cpu().numpy() 

    if len(detections) == 0:
        st.warning("No traffic signs detected in the image.")
    else:
        for i, detection in enumerate(detections):
            x_min, y_min, x_max, y_max = detection[:4].astype(int)
            cropped_image = img_array[y_min:y_max, x_min:x_max]

            cropped_image_resized = cv2.resize(cropped_image, (90, 90))
            cropped_image_resized = cropped_image_resized / 255.0
            cropped_image_resized = np.expand_dims(cropped_image_resized, axis=0)

            # Predict using CNN
            prediction = cnn_model.predict(cropped_image_resized)
            predicted_class = np.argmax(prediction)
            print(predicted_class)
            predicted_class_name = class_names[predicted_class]
            confidence = np.max(prediction) 

            cv2.rectangle(
                img_array,
                (x_min, y_min), 
                (x_max, y_max),  
                color=(0, 255, 0), 
                thickness=2
            )

            label = f"{predicted_class_name}, {confidence:.2f}"

            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            if y_min - text_height - 10 < 0:  
                label_y = y_max + 10 
            else:
                label_y = y_min - 5  

            if x_max + text_width > img_array.shape[1]:  
                label_x = x_min - text_width
                if label_x < 0: 
                    label_x = x_min + 10
            else:
                label_x = x_min 

            cv2.rectangle(
                img_array,
                (label_x, label_y - text_height - 10), 
                (label_x + text_width, label_y), 
                (0, 255, 0), 
                -1
            )

            cv2.putText(
                img_array,
                label,  
                (label_x, label_y - 5),  
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                1  
            )


        st.markdown(
            """
            <div style="display: flex; justify-content: center;">
                <img src="data:image/png;base64,{}" width="450" />
            </div>
            """.format(image_to_base64(img_array)),
            unsafe_allow_html=True
        )

else:
    st.info("Please upload an image to get started.")
