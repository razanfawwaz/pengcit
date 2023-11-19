import streamlit as st
import cv2
import numpy as np
from multipage_streamlit import State

# Define a multipage app
state = State(__name__)


# Function to get the chain code of a binary image
def get_chain_code(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    contour = contours[0]  # Assuming only one contour for simplicity
    
    chain_code = []
    directions = [0, 1, 2, 3, 4, 5, 6, 7]

    for point in contour:
        x, y = point[0]
        chain_code.append(directions.index((x + 1, y) in contour))
    
    return chain_code

# Page 1: Image Upload and Chain Code
def main():
    st.title("Upload Image for Chain Code")

    # Upload an image
    uploaded_file = st.file_uploader("Choose a binary image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        binary_img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        st.image(binary_img, caption="Uploaded Binary Image", use_column_width=True)

        # Get and display the chain code
        chain_code = get_chain_code(binary_img)

        if chain_code is not None:
            st.subheader("Chain Code:")
            st.write(chain_code)
        else:
            st.warning("No contour found in the binary image.")

# Run the app
state.save()
