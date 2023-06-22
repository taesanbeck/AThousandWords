# page_2.py
import streamlit as st
from PIL import Image

def show_page(selectbox):
    st.title('User Application')

    st.header('Upload an image:')
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=500)

        st.header('Confidence Level:')
        confidence_level = st.slider('Adjust the confidence level', min_value=0.0, max_value=1.0, value=0.5)

        st.header('Bounding Boxes:')
        bounding_box_option = st.radio('Would you like bounding boxes displayed?', ('Yes', 'No'))

    



