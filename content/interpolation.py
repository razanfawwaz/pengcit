import streamlit as st
import cv2
import numpy as np
from multipage_streamlit import State

state = State(__name__)

# Function to perform image interpolation
def perform_interpolation(image, interpolation_method, new_width, new_height):
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation_method)

# Streamlit app
def main():
    st.title("Image Interpolation App")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), -1)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        new_width = st.number_input("Enter New Width:", value=image.shape[1])
        new_height = st.number_input("Enter New Height:", value=image.shape[0])

        interpolation_methods = {
            'Nearest Neighbor': cv2.INTER_NEAREST,
            'Linear': cv2.INTER_LINEAR,
            'Cubic': cv2.INTER_CUBIC,
            'Area': cv2.INTER_AREA,
            # Add more methods here if needed
        }
        selected_interpolation = st.selectbox("Select Interpolation Method:", list(interpolation_methods.keys()))

        if st.button("Interpolate"):
            result = perform_interpolation(image, interpolation_methods[selected_interpolation], new_width, new_height)
            st.image(result, caption=f"Interpolated Image ({selected_interpolation})", use_column_width=True)

state.save()