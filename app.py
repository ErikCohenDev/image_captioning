import os
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
import json

st.title("Welcome to Image Captioner")
st.header("Get Captions for your photos")

LOSS_PROGRESSION_DATASET = [4.1515,
3.4366,
3.2091,
3.0741,
2.9780,
2.9009,
2.8432,
2.7974,
2.7573,
2.7225,
2.6943,
2.6654,
2.6459,
2.6236,
2.6047,
2.5895,
2.5711,
2.5590,
2.5471,
2.5341,
2.5268,
2.5170,
2.5093,
2.4979,
2.4913,
2.4826,
2.4743,
2.4683,
2.4623,
2.4571,
2.4496,
2.4488,
2.4399,
2.4346,
2.4289,
2.4284,
2.4247,
2.4195,
2.4178,
2.4118,
2.4120,
2.4067,
2.4019,
2.3986,
2.3983,
2.3932,
2.3925,
2.3889,
2.3834,
2.3804,
2.3830,
2.3807,
2.3779,
2.3765,
2.3747,
2.3675,
2.3720,
2.3698,
2.3656,
2.3610]

def generate_caption(image):
    from main import make_prediction
    st.session_state.caption = make_prediction(image)

def clear_caption_state():
    st.session_state.caption = None

def main():
    uploaded_file = st.file_uploader("Choose a file", type='jpeg,jpg,png', on_change=clear_caption_state)
    if uploaded_file is not None:
        # To read file as bytes:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        if st.session_state.caption:
            st.caption(f"<h2 style='text-align:center'>{st.session_state.caption}</h2>", unsafe_allow_html=True)

        st.button('Predict', on_click=lambda:generate_caption(image))
    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:white;" /> """, unsafe_allow_html=True)
    st.header("Training vs Validation Data Chart")
    chart_data = pd.DataFrame(
     [[0,6000], [2000, 0]],
     columns=["Testing", "Training"]
    )
    st.bar_chart(chart_data)
    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:white;" /> """, unsafe_allow_html=True)
    st.header("Linear chart")
    st.text("The Graph shows the model loss decreasing as the model gets trained \nthrough more EPOCHS")
    st.line_chart(data=LOSS_PROGRESSION_DATASET)
    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:white;" /> """, unsafe_allow_html=True)
    st.header("Scatter Distribution Plot")
    st.text("Each point represents the BLEU Score when comparing the predicted result from \nthe test dataset and the actual captions for the photo.")
    with open('bleu-scores.json', 'r') as fp:
        bleu_scores = json.load(fp)
        df = pd.DataFrame.from_dict(bleu_scores)
        df['bleu_score'] = df['bleu_score'].multiply(10).round(decimals = 3)
        df = df.dropna(subset=['bleu_score'])
        bins = [2, 3, 4, 5, 6, 7, 8]
        labels = [2, 3, 4, 5, 6, 7]
        df['binned'] = pd.cut(df['bleu_score'], bins=bins, labels=labels)
        scatterplot = alt.Chart(df).mark_circle(size=50).encode(
            x='bleu_score',
            y=alt.Y('binned:N', axis=alt.Axis(labels=False)),
            color=alt.Color('binned:N', legend=None)
        ).interactive().properties(
            width=600,
            height=300
        )
        st.write(scatterplot)

main()
