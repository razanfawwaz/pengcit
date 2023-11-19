import streamlit as st
import cv2
import numpy as np

def restore_image(input_image, restoration_method):
    if restoration_method == "Noise Reduction":
        # Implementasi pengurangan noise (contoh menggunakan filter Gaussian)
        restored_image = cv2.GaussianBlur(input_image, (5, 5), 0)
    elif restoration_method == "Bluring":
        # Implementasi blurring (contoh menggunakan filter averaging)
        kernel_size = 5
        restored_image = cv2.blur(input_image, (kernel_size, kernel_size))
    else:
        # Jika metode restorasi tidak dipilih, kembalikan gambar asli
        restored_image = input_image.copy()

    return restored_image

def main():
    st.title("Image Restoration with Streamlit")

    # Upload gambar dari pengguna
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Baca gambar
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, 1)

        # Tampilkan gambar asli
        st.image(original_image, caption="Original Image", use_column_width=True)

        # Pilihan metode restorasi
        restoration_method = st.selectbox("Choose Restoration Method", ["None", "Noise Reduction", "Bluring"])

        if restoration_method != "None":
            # Restorasi gambar
            restored_image = restore_image(original_image, restoration_method)
            st.image(restored_image, caption="Restored Image", use_column_width=True)

if __name__ == "__main__":
    main()
