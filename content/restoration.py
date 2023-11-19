import streamlit as st
import cv2
import numpy as np
from multipage_streamlit import State

# Define a multipage app
state = State(__name__)

# Function to apply image restoration techniques
def apply_lowpass_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def apply_median_filter(image):
    return cv2.medianBlur(image, 5)

def apply_rank_order_filter(image):
    return cv2.boxFilter(image, -1, (5, 5))

def apply_outlier_method(image):
    return cv2.bilateralFilter(image, 9, 75, 75)

# Page 1 & 2: Image Upload and Restoration

def main():
    st.title("Image Restoration Techniques")

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        # Convert the image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Display the original grayscale image
        st.image(img_gray, caption="Original Grayscale Image", use_column_width=True)

        # Apply restoration techniques
        st.subheader("Restoration Techniques")

        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        with col1:
            st.write("Lowpass Filter")
            restored_lowpass = apply_lowpass_filter(img_gray)
            st.image(restored_lowpass, caption="Lowpass Filter Result", use_column_width=True)

        with col2:
            st.write("Median Filter")
            restored_median = apply_median_filter(img_gray)
            st.image(restored_median, caption="Median Filter Result", use_column_width=True)
        
        with col3:
            st.write("Rank-order Filter")
            restored_rank_order = apply_rank_order_filter(img_gray)
            st.image(restored_rank_order, caption="Rank-order Filter Result", use_column_width=True)

        with col4:
            st.write("Outlier Method")
            restored_outlier = apply_outlier_method(img_gray)
            st.image(restored_outlier, caption="Outlier Method Result", use_column_width=True)

# Run the app
state.save()
