import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from multipage_streamlit import State

# Initialize the state
state = State(__name__)


def remove_black_color(image):
    """Segments the black color from an image."""

    # Convert the image to HSV color space.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the black color in HSV color space.
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 255])

    # Create a mask for the black color.
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Set the black pixels to transparent in the original image.
    image[mask] = [0, 0, 0]

    # Return the segmented image.
    return image


def main():
    st.title("Remove Black Color")

    # Upload an image.
    uploaded_file = st.file_uploader("Upload an image:")

    # If an image is uploaded, segment the black color and display the result.
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image = cv2.imdecode(np.fromstring(image_bytes, np.uint8), 1)
        # Display the original and segmented images.
        st.image(image, caption="Original image", channels="BGR")

        segmented_image = remove_black_color(image)

        st.image(segmented_image, caption="Segmented image", channels="BGR")


state.save()
