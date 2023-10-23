import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from multipage_streamlit import State

# Initialize the state
state = State(__name__)

# Function to apply a cartoon effect using a pre-trained U-Net model


def cartoonize_with_model(image):
    # Load the pre-trained U-Net model for image segmentation
    model = load_model('cartoonization_model.h5')

    # Resize the image to a fixed size (e.g., 256x256) required by the model
    image = cv2.resize(image, (256, 256))

    # Normalize the image to values between 0 and 1
    image = image / 255.0

    # Predict the segmentation mask using the model
    mask = model.predict(np.expand_dims(image, axis=0))[0]

    # Threshold the mask to create a binary mask
    mask[mask >= 0.5] = 1.0
    mask[mask < 0.5] = 0.0

    # Apply a bilateral filter to smooth the image and create a cartoon effect
    cartoon = cv2.stylization(image, sigma_s=150, sigma_r=0.25)

    # Apply the binary mask to the cartoon image to retain only the edges
    cartoon = cartoon * mask

    # Convert the cartoon image back to the original scale
    cartoon = (cartoon * 255).astype(np.uint8)

    return cartoon


def main():
    # Streamlit app
    st.title("Face to Cartoon Streamlit App")

    uploaded_image = st.file_uploader(
        "Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image_bytes = uploaded_image.getvalue()
        image = cv2.imdecode(np.fromstring(image_bytes, np.uint8), 1)
        st.image(image, caption="Original Image",
                 use_column_width=True, channels="BGR")

        if st.button("Cartoonize"):
            cartoon_image = cartoonize_with_model(image)
            st.image(cartoon_image, caption="Cartoonized Image",
                     use_column_width=True)


state.save()
