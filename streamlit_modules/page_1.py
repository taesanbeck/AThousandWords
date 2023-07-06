from objects.yolo8 import standalone_yolo, output_class_list
from PIL import Image
import streamlit as st
from tts.texttospeech import texttospeech  
from nlp.t5_coco import generate_caption, model_instance
from objects.yolo3 import run_yolo3
from objects.yolo8 import run_yolo8
from nlp.t5_coco import run_t5
import io
import os

def show_page(selected_cv_model, selected_nlp_model):
    st.title('Model Testing')

    st.header('Upload an image:')
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    st.header('Confidence Level:')
    confidence_level = st.slider('Adjust the confidence level', min_value=0.0, max_value=1.0, value=0.5)

    st.header('Bounding Boxes:')
    bounding_box_option = st.radio('Would you like bounding boxes displayed?', ('Yes', 'No'))

    # Add a button to run the model and generate a caption
    if st.button('Run Models'):
        if selected_cv_model == 'YOLOV8':
            labels = run_yolo8(uploaded_file, bounding_box_option, confidence_level)
            
        elif selected_cv_model == 'YOLOV3':
            # Pass all three required arguments to the run_yolo3 function
            image_input = Image.open(uploaded_file)
            image_name = uploaded_file.name
            labels = run_yolo3(image_input, image_name, confidence_level)

        if selected_nlp_model == 'T5':
            run_t5(labels)
            
        elif selected_nlp_model == 'GPT2':
            # Add code here to generate a caption using GPT2
                pass








    
    

