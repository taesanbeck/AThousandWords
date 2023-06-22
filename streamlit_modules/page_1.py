# page_1.py
import streamlit as st
from PIL import Image
from objects import yolo, dino

def show_page(selected_model):
    st.title('Model Testing')

    st.header('Upload an image:')
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    st.header('Confidence Level:')
    confidence_level = st.slider('Adjust the confidence level', min_value=0.0, max_value=1.0, value=0.5)

    st.header('Bounding Boxes:')
    bounding_box_option = st.radio('Would you like bounding boxes displayed?', ('Yes', 'No'))

    # Load the chosen model
    if selected_model == 'Yolo':
        model = yolo()
    elif selected_model == 'DINO':
        model = dino()

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=500)

        # Replace "model_prediction" with model's prediction function
        cv_labels = model.predict(image, confidence_level, bounding_box_option)

        # Display the labels
        st.header('Computer Vision Labels:')
        st.text(cv_labels)

        # Pass the labels to  NLP model for description
        # nlp_description = nlp_model.describe(cv_labels)

        # Display the description
        # st.header('NLP Description:')
        # st.text(nlp_description)