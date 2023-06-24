# page_1.py
import streamlit as st
from PIL import Image
from objects import yolo, dino
from objects.yolo import standalone_yolo, output_class_list

standalone = True # if we can't do sagemaker for some reason, otherwise set to false


def show_page(selected_model):
    st.title('Model Testing')

    st.header('Upload an image:')
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    st.header('Confidence Level:')
    confidence_level = st.slider('Adjust the confidence level', min_value=0.0, max_value=1.0, value=0.5)

    st.header('Bounding Boxes:')
    bounding_box_option = st.radio('Would you like bounding boxes displayed?', ('Yes', 'No'))

    if bounding_box_option == 'Yes':
        bounding_box_option = True
    if bounding_box_option == 'No':
        bounding_box_option = False


    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=500)

        # Replace "model_prediction" with model's prediction function
        # there is an option to remove bounding boxes from the image output of Yolo, but there is no point
        # just show the original if you don't want bounding boxes

        if selected_model == 'YOLO':
            if standalone == True:
                output_dict = standalone_yolo(image, confidence=confidence_level, save_img=bounding_box_option)

        #if selected_model == 'DINO':
            # do something else

        labels = output_class_list([output_dict])

        # Display the labels
        st.header('Computer Vision Labels:')
        st.text(labels)

        # Pass the labels to  NLP model for description
        # nlp_description = nlp_model.describe(labels)

        # Display the description
        # st.header('NLP Description:')
        # st.text(nlp_description)