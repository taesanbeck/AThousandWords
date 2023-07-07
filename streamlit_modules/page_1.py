from objects.yolo8 import output_class_list, output_class_list_w_meta
from PIL import Image
import streamlit as st
from tts.texttospeech import texttospeech  
#from nlp.t5_coco import generate_caption, model_instance
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

        if uploaded_file is not None: # open one time only
            image_input = Image.open(uploaded_file)
            image_name = uploaded_file.name

        if selected_cv_model == 'YOLOV8':
            # get the labels
            raw_results = run_yolo8(image_input, image_name, bounding_box_option, confidence_level)
            labels = output_class_list_w_meta(raw_results)
            # this needs to be like this because additional models will be adding to raw_results before the final
            # t5-ready labels are made
            # tl;dr I'm going to need this later so please leave it alone
            
        elif selected_cv_model == 'YOLOV3':
            # Pass all three required arguments to the run_yolo3 function
            labels = run_yolo3(image_input, image_name, confidence_level, bounding_box_option)

        if selected_nlp_model == 'T5':
            run_t5(labels)
            
        elif selected_nlp_model == 'GPT2':
            # Add code here to generate a caption using GPT2 or other model
                pass









    
    

