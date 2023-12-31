import streamlit as st
import cv2
import numpy as np

def chaincode(image):
    # Grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    st.image(gray_image, caption="Grayscale", use_column_width=True)

    # Thresholding
    _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Cari kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Ambil kontur pertama
    contour = contours[0]

    # Chain code
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx_chain = cv2.approxPolyDP(contour, epsilon, True)

    st.write("Chain Code:", approx_chain)

def main():
    st.title("Chain Code Calculation with Streamlit")

    # Upload gambar dari pengguna
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Baca gambar
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, 1)

        # Tampilkan gambar asli
        st.image(original_image, caption="Original Image", use_column_width=True)

        # Jalankan chain code
        chaincode(original_image)

if __name__ == "__main__":
    main()
