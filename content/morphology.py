import streamlit as st
import cv2
import numpy as np
from multipage_streamlit import State

state = State(__name__)

# Function to apply morphological operations
def apply_morphology(image, operation, kernel):
    if operation == "Dilation":
        return cv2.dilate(image, kernel, iterations=1)
    elif operation == "Erosion":
        return cv2.erode(image, kernel, iterations=1)
    elif operation == "Opening":
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == "Closing":
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    elif operation == "Morph Gradient":
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

def main():
    # Streamlit app
    st.title("Image Morphology Operations")

    # Upload an image
    image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if image is not None:
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
        
        # Convert the image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        st.image(img_gray, caption="Grayscale Image", use_column_width=True)

        # Sidebar controls
        st.sidebar.header("Morphological Operations")
        operation = st.sidebar.selectbox("Select Operation", ["Dilation", "Erosion", "Opening", "Closing", "Morph Gradient"])
        kernel_size = st.sidebar.slider("Kernel Size", min_value=1, max_value=21, step=2)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply the selected operation
        if st.button("Apply"):
            if img_gray is not None:
                result = apply_morphology(img_gray, operation, kernel)
                st.image(result, caption=operation + " Result", use_column_width=True)