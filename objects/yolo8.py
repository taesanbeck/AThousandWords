from ultralytics import YOLO
from PIL import Image
import os
import cv2
from utils.location import find_bbox_centroid, find_quadrant
import numpy as np

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



from PIL import Image
import streamlit as st

def run_yolo8(uploaded_file, bounding_box_option, confidence_level):
    if uploaded_file is not None:
        image_input = Image.open(uploaded_file)
        image_name = uploaded_file.name
        
        # It would be a good idea to check and see if predict_filename is already in saved_img
        # if it is, don't run, just show the image.
        # however, we would also need to save the results dict, possibly as json, so we can call it back up.

        if bounding_box_option == 'Yes':
            results, image_output = standalone_yolo(image_input, image_name=image_name,
                                                    confidence=confidence_level, save_img=True)
        if bounding_box_option == 'No':
            results, image_output = standalone_yolo(image_input, image_name=image_name,
                                                    confidence=confidence_level, save_img=False)

        st.image(image_output, caption='Uploaded Image', use_column_width=True)  # Display the uploaded image

        # watch this
        labels = output_class_list_w_meta(results)
        
        return labels








