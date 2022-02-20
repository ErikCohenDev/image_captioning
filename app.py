import os
import streamlit as st
from PIL import Image
import pandas as pd

from main import make_prediction

# Setup environment credentials (you'll need to change these)
# change for your GCP key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "decent-genius-341102-96e86cea1f7c.json"
PROJECT = "decent-genius-341102"  # change for your GCP project
# change for your GCP region (where your model is hosted)
REGION = "us-central1"

st.title("Welcome to Image Captioner")
st.header("Get Captions for your photos")

MODEL = "V1 Model"

def generate_caption(image):
    st.session_state.caption = make_prediction(image)

def clear_caption_state():
    st.session_state.caption = None

def main():
    uploaded_file = st.file_uploader("Choose a file", on_change=clear_caption_state)
    if uploaded_file is not None:
        # To read file as bytes:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        if st.session_state.caption:
            st.caption(f"<h2 style='text-align:center'>{st.session_state.caption}</h2>", unsafe_allow_html=True)

        st.button('Predict', on_click=lambda:generate_caption(image))

main()
