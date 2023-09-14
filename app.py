import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set the title and description of the Streamlit app
st.title("Image Histogram Normalization")
st.write("Upload an image and adjust the clip limit for histogram equalization.")

# Upload an image using Streamlit's file uploader
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Add a slider to set the clip limit for histogram equalization
clip_limit = st.slider("Equalization Clip Limit", min_value=1, max_value=10, value=2)


def plot_histograms(original_image, processed_image):
    original_hist = cv2.calcHist([original_image], [0], None, [256], [0, 256])
    processed_hist = cv2.calcHist([processed_image], [0], None, [256], [0, 256])

    fig, ax = plt.subplots(2, 1, figsize=(8, 10))  # Increase the figsize here
    ax[0].set_title("Original Image Histogram")
    ax[0].plot(original_hist)
    ax[1].set_title("Equalized Image Histogram")
    ax[1].plot(processed_hist)

    return fig


if uploaded_image is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform histogram equalization with the clip_limit parameter
    equalized_image = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(gray_image)
    colored_equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

    # Display the slider, images, and histograms in three columns
    col1, col2 = st.columns(2)

    # Display original image in the first column
    with col1:
        st.image(image, caption="Original Image", use_column_width=True, channels="BGR")

    # Display processed image in the second column
    with col2:
        st.image(colored_equalized_image, caption="Processed Image", use_column_width=True, channels="BGR")

    st.pyplot(plot_histograms(gray_image, equalized_image))
