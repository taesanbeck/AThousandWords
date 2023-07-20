from objects.yolo8 import output_class_list, output_class_list_w_meta, run_yolo8
from objects.yolo3 import run_yolo3
from PIL import Image
import streamlit as st
from tts.texttospeech import texttospeech  
from nlp.t5_coco import generate_caption, model_instance, run_t5
from nlp.t5_common_gen import run_t5_common_gen
import io
import os
from scenes.densenet import run_densenet
import traceback

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
        if uploaded_file is not None:
            image_input = Image.open(uploaded_file)
            image_name = uploaded_file.name

        try:
            labels = None
            raw_results = None
            if selected_cv_model == 'YOLOV8':
                raw_results = run_yolo8(image_input, image_name, bounding_box_option, confidence_level)
                labels = output_class_list_w_meta(raw_results)
                #if 'person' in output_class_list(raw_results):
                    #run yolo actions, output updated labels
            elif selected_cv_model == 'YOLOV3':
                labels = run_yolo3(image_input, image_name, confidence_level, bounding_box_option)

            #run scene recognition, output updated labels
            labels, raw_results = run_densenet(raw_results, labels, image_input)

            #run ocr, output updated labels

            if not labels:
                st.error('No objects detected in the uploaded image.')
            else:
                # put any preprocessing here
                if selected_nlp_model == 'T5':
                    run_t5(labels)
                elif selected_nlp_model == 'T5_Common_Gen':
                    run_t5_common_gen(labels)
        except Exception as e:
            st.error(f'Error: {traceback.format_exc()}')









    
    

