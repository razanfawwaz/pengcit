import streamlit as st
import cv2
import numpy as np

def chaincode(image):
    # Grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    st.image(gray_image, caption="Grayscale", use_column_width=True)

    # Deteksi tepi
    edge_image = cv2.Canny(gray_image, 100, 200)
    st.image(edge_image, caption="Edge Detection", use_column_width=True)

    # Dilatasi
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(edge_image, kernel, iterations=1)
    st.image(dilated_image, caption="Dilated Image", use_column_width=True)

    # Erosi
    eroded_image = cv2.erode(dilated_image, kernel, iterations=1)
    st.image(eroded_image, caption="Eroded Image", use_column_width=True)

    # Contoh lainnya: menambahkan operasi pengolahan citra sesuai kebutuhan Anda
    # ...

def main():
    st.title("Image Processing Chaincode with Streamlit")

    # Upload gambar dari pengguna
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Baca gambar
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, 1)

        # Tampilkan gambar asli
        st.image(original_image, caption="Original Image", use_column_width=True)

        # Jalankan chaincode
        chaincode(original_image)

if __name__ == "__main__":
    main()
