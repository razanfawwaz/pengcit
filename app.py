import streamlit as st
from multipage_streamlit import State
from content import histogram, faceblur, edge, sharpening, morphology, interpolation

# Initialize the state
state = State(__name__)

# Create a dictionary to map page names to their corresponding functions
pages = {
    "Histogram Normalization": histogram.main,
    "Face Blurring": faceblur.main,
    "Edge Detection": edge.main,
    "Sharpening": sharpening.main,
    "Morphology": morphology.main,
    "Interpolation": interpolation.main,
}

st.sidebar.title("Pengolahan Citra")
st.sidebar.caption("Tugas Pengolahan Citra B - Kelompok 6")
# Add a navigation sidebar to switch between pages
page = st.sidebar.selectbox("Select a Page", list(pages.keys()))

# Call the selected page function
pages[page]()

# Save the state
state.save()
