import streamlit as st
import cv2
import numpy as np
from multipage_streamlit import State

# Initialize the state
state = State(__name__)


def main():
    # Set the title and description of the Streamlit app
    st.title("Face Blurring App")
    st.write("Upload an image, and we'll blur the faces in it.")

    # Upload an image using Streamlit's file uploader
    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "png", "jpeg"])

    slider = st.slider("Select a value", 10, 100, 50)

    if uploaded_image is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.frombuffer(
            uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

        # Load the pre-trained Haar Cascade classifier for face detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert the image to grayscale for face detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Create a Streamlit column layout for "Before" and "After" images
        col1, col2 = st.columns(2)

        # Display the original image labeled as "Before"
        with col1:
            st.image(image, caption="Before",
                     use_column_width=True, channels="BGR")

        # Process the image to blur faces
        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) of each face
            face_roi = image[y:y + h, x:x + w]

            # Apply Gaussian blur to the ROI to blur the face
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), slider)

            # Replace the original face with the blurred face
            image[y:y + h, x:x + w] = blurred_face

        # Display the modified image labeled as "After"
        with col2:
            st.image(image, caption="After",
                     use_column_width=True, channels="BGR")


state.save()
