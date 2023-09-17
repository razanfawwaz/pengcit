import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from multipage_streamlit import State

# Initialize the state
state = State(__name__)

# Define a function to plot the histograms


def plot_histograms(image):
    image_hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))  # Increase the figsize here
    ax.set_title("Image Histogram")
    ax.plot(image_hist)

    # Do not display the histogram in the Streamlit app
    st.pyplot(fig)


# Define functions for each page


def main():
    st.header("Histogram Normalization")
    st.title("Image Histogram Normalization")
    st.write("Upload an image for histogram equalization.")

    # Upload an image using Streamlit's file uploader
    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.frombuffer(
            uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform histogram equalization without specifying clip limit
        equalized_image = cv2.equalizeHist(gray_image)
        colored_equalized_image = cv2.cvtColor(
            equalized_image, cv2.COLOR_GRAY2BGR)

        # Display the slider, images, and histograms in three columns
        col1, col2 = st.columns(2)

        # Display original image in the first column
        with col1:
            st.image(image, caption="Original Image",
                     use_column_width=True, channels="BGR")

        # Display processed image in the second column
        with col2:
            st.image(colored_equalized_image, caption="Processed Image",
                     use_column_width=True, channels="BGR")

        col3, col4 = st.columns(2)

        with col3:
            st.write("Original Image Histogram")
            plot_histograms(gray_image)
        with col4:
            st.write("Processed Image Histogram")
            plot_histograms(equalized_image)


state.save()
