import streamlit as st
import cv2
import numpy as np
from multipage_streamlit import State

# Initialize the state
state = State(__name__)


def main():
    # Set the title and description of the Streamlit app
    st.title("Image Processing App")
    st.write("Upload an image and choose an operation.")

    # Upload an image using Streamlit's file uploader
    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.frombuffer(
            uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

        # Convert the image to grayscale for edge detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray_image, 100, 200)

        # Display the edge-detected image
        st.image(edges, caption="Edge Detection",
                 use_column_width=True, channels="GRAY")


state.save()
