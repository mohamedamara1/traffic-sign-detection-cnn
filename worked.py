import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2

# Functions for preprocessing
def resize_image(image, size=(640, 640)):
    return cv2.resize(image, size)

def normalize_image(image):
    return image / 255.0

# Load the YOLO model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

def draw_detections(image, boxes, class_names):
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())
        label = f"{class_names[cls]}: {conf:.2f}"

        cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (xyxy[0], xyxy[1] - text_height - 10), 
                      (xyxy[0] + text_width, xyxy[1]), (0, 255, 0), -1)
        cv2.putText(image, label, (xyxy[0], xyxy[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return image
def scale_boxes(boxes, original_size, new_size):
    scale_x = original_size[1] / new_size[1]
    scale_y = original_size[0] / new_size[0]
    
    scaled_boxes = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.xyxy[0].cpu().numpy().astype(int)
        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)
        scaled_boxes.append([x1, y1, x2, y2, conf, cls])
        
    return scaled_boxes

# Title and description
st.title("Traffic Sign Detection")
st.write("Upload an image to detect traffic signs using YOLOv8.")

# Upload section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Load model and classes
model_path = "best.onnx"
model = load_model(model_path)
class_names = model.names
st.write("Model Classes:", model.names)

if uploaded_file:
    # Read and preprocess the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize and normalize the image
    resized_image = resize_image(image, size=(640, 640))
    normalized_image = normalize_image(resized_image)
    normalized_image_uint8 = (normalized_image * 255).astype(np.uint8)

    # Display the original image
    st.subheader("Original Image")
    st.image(image, channels="RGB")

    # Run inference with resized and normalized image
    st.subheader("Running detection...")
    results = model.predict(source=normalized_image_uint8, imgsz=640, conf=0.5)

    # Extract bounding boxes
    boxes = results[0].boxes if results and len(results) > 0 else None

    if boxes and len(boxes) > 0:
        st.subheader("Detection Results")
        
        detected_classes = [class_names[int(box.cls[0].cpu().numpy())] for box in boxes]
        detected_confidences = [box.conf[0].cpu().numpy() for box in boxes]
        st.write({"Class": detected_classes, "Confidence": [f"{conf:.2f}" for conf in detected_confidences]})

        # Draw detections on the image
        result_image = draw_detections(image.copy(), boxes, class_names)

        # Display the image with detections
        st.image(result_image, channels="RGB", caption="Detected Traffic Signs")
    else:
        st.write("No traffic signs detected.")
