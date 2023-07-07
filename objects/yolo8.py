from ultralytics import YOLO
from PIL import Image
import os
import cv2
from utils.location import find_bbox_centroid, find_quadrant
import numpy as np
import streamlit as st

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def standalone_yolo(image, confidence, save_img, image_name):
    model=YOLO('yolov8x.pt') # will download the model if it isn't already there
    detection = model.predict(image, conf=confidence)
    output = [{'class': box.cls.item(), 'class_name': detection[0].names[box.cls.item()],
               'xyxy': box.xyxy.tolist()[0], 'conf': box.conf.item()} for box in detection[0].boxes]
    i = 1
    while i < len(output):
        for object in output:
            object.update({'id': str(i)})
            i+=1
    if save_img == False:
        result_image = image
    if save_img == True:
        if not os.path.exists(os.path.join('objects', 'saved_img')):
            os.mkdir(os.path.join('objects', 'saved_img'))
        raw_output = detection[0].plot(pil=True)
        cv2.imwrite(os.path.join('objects', 'saved_img', 'predict_'+image_name), raw_output)
        result_image = Image.open(os.path.join('objects', 'saved_img', 'predict_'+image_name))

    for object in output:
        centroid = find_bbox_centroid(object['xyxy'][0], object['xyxy'][1], object['xyxy'][2], object['xyxy'][3])
        location = find_quadrant(image, centroid[0], centroid[1])
        object.update({'location': location})

    return output, result_image

def output_class_list(olist):
    # get only the class predictions as human readable names
    return [o['class_name'] for o in olist]

def output_class_list_w_meta(olist):
    # return human readable names plus fun stuff
    return [o['id']+': '+o['class_name']+' @ '+o['location'] for o in olist]



######## HERE IS THE RUN MODEL Function for you

def run_yolo8(uploaded_file, selected_cv_model, bounding_box_option, confidence_level, selected_nlp_model):
    labels = []  # Initialize labels as an empty list
    if uploaded_file is not None:
        image_input = Image.open(uploaded_file)
        image_name = uploaded_file.name

        if selected_cv_model == 'YOLOV3':
            # Call the run_yolo3 function and pass the necessary arguments
            labels = run_yolo8(image_input, image_name, confidence_level, bounding_box_option)
            
        elif selected_cv_model == 'YOLOV8':
            # Run the YOLO model on the image
            if bounding_box_option == 'Yes':
                results, image_output = standalone_yolo(image_input, confidence=confidence_level, save_img=True, image_name=image_name)
            if bounding_box_option == 'No':
                results, image_output = standalone_yolo(image_input, confidence=confidence_level, save_img=False, image_name=image_name)

            st.image(image_output, caption='Uploaded Image', use_column_width=True)  # Display the uploaded image

            # watch this
            labels = output_class_list_w_meta(results)
    
    return labels  # Return labels as a list














