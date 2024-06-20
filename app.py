
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Load the YOLOv8 model
model = YOLO('E:/runs/detect/train/weights/best.pt')

def detect_objects(image):
    # Convert image to OpenCV format (numpy array)
    img_np = np.array(image)
    
    # Run prediction
    results = model(img_np)
    
    # Draw bounding boxes on the image
    annotated_img = img_np.copy()
    
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0]
        conf = result.conf[0]
        cls = result.cls[0]

        # Rescale bounding box coordinates to match the resized image
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
        cv2.putText(annotated_img, f'{model.names[int(cls)]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return annotated_img

def main():
    st.title('YOLOv8 Object Detection App')
    st.write("Upload an image and see object detection results!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Read image file
        image = Image.open(uploaded_file)

        # Perform object detection and annotation
        annotated_image = detect_objects(image)

        # Display original and annotated image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.image(annotated_image, caption='Annotated Image', use_column_width=True)

if __name__ == '__main__':
    main()
