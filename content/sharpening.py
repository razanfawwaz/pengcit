import streamlit as st
from PIL import Image, ImageOps, ImageFilter



def sharpen_image(image):
    # Melakukan sharpening pada gambar menggunakan filter sharpen
    sharpened_image = image.filter(ImageFilter.SHARPEN)
    return sharpened_image

def rescale_image(image, scale_factor):
    # Merescale gambar dengan faktor tertentu
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return image.resize((new_width, new_height))

def convert_to_grayscale(image):
    # Mengubah gambar ke grayscale
    grayscale_image = ImageOps.grayscale(image)
    return grayscale_image

def main():
    st.title("Sharpening, Scaling, Grayscale")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Original Image", use_column_width=True)

        # Sharpening gambar
        sharpened_image = sharpen_image(original_image)
        st.image(sharpened_image, caption="Sharpened Image", use_column_width=True)

        # Menyimpan gambar sharpened
        if st.button("Save Sharpened Image"):
            save_path = "sharpened_image.png"
            sharpened_image.save(save_path)
            st.success(f"Sharpened Image saved as {save_path}")

        scale_factor = st.slider("Scale Factor", 0.1, 2.0, 1.0, 0.1)
        # Merescale gambar
        rescaled_image = rescale_image(sharpened_image, scale_factor)
        st.image(rescaled_image, caption=f"Rescaled Image (Factor: {scale_factor})", use_column_width=True)

        # Menyimpan gambar rescaled
        if st.button("Save Rescaled Image"):
            save_path = "rescaled_image.png"
            rescaled_image.save(save_path)
            st.success(f"Rescaled Image saved as {save_path}")

        # Mengubah gambar menjadi grayscale
        grayscale_image = convert_to_grayscale(rescaled_image)
        st.image(grayscale_image, caption="Grayscale Image", use_column_width=True)

        # Menyimpan gambar grayscale
        if st.button("Save Grayscale Image"):
            save_path = "grayscale_image.png"
            grayscale_image.save(save_path)
            st.success(f"Grayscale Image saved as {save_path}")
