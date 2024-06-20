# from ultralytics import YOLO
# import cv2

# # Load the trained YOLOv8 model
# model = YOLO('E:/runs/detect/train/weights/best.pt')

# # Load an image
# img = cv2.imread('E:/CV/datacluster_000040.jpg')

# # Check if the image is loaded correctly
# if img is None:
#     print("Error: Image not found or unable to read.")
#     exit()

# # Run prediction
# results = model(img)

# # Print results
# print(results)

# # Resize the image for better display (adjust the size as needed)
# scale_percent = 20  # percent of original size
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)

# # Resize image
# resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# # Check if any boxes are detected
# if len(results[0].boxes) == 0:
#     print("No detections found.")

# # Display the image with bounding boxes
# for result in results[0].boxes:
#     x1, y1, x2, y2 = result.xyxy[0]
#     conf = result.conf[0]
#     cls = result.cls[0]

#     # Rescale bounding box coordinates to match the resized image
#     x1, y1, x2, y2 = int(x1 * scale_percent / 100), int(y1 * scale_percent / 100), int(x2 * scale_percent / 100), int(y2 * scale_percent / 100)

#     cv2.rectangle(resized_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#     cv2.putText(resized_img, f'{model.names[int(cls)]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# cv2.imshow('Prediction', resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


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

        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
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
