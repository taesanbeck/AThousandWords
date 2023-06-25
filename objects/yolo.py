from ultralytics import YOLO

def standalone_yolo(image, confidence, save_img=False):
    model=YOLO('https://thousandwordsgmu.s3.amazonaws.com/yolov8x.pt') # will download the model if it isn't already there
    detection = model(image, conf=confidence, save=save_img, project='objects', name='saved_img')
    output = [{'class': box.cls.item(), 'class_name': detection[0].names[box.cls.item()],
               'xyxy': box.xyxy.tolist()[0], 'conf': box.conf.item()} for box in detection[0].boxes]

    return output

def output_class_list(olist):  # get only the class predictions as human readable names

    return [o['class_name'] for o in olist]


from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
from ultralytics.yolo.utils.plotting import Annotator
from typing import Dict, Union, List, Tuple
import streamlit as st
from dotenv import load_dotenv
import os
from smart_open import open

# Load environment variables from .env file
load_dotenv()

# Retrieve AWS credentials from environment variables
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')


# Load environment variables from .env file
load_dotenv()

# Retrieve AWS_SECRET_ACCESS_KEY from environment variables
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')



# https://stackoverflow.com/questions/75324341/yolov8-get-predicted-bounding-box

def standalone_yolo2(image: Image.Image, confidence: float) -> Tuple[Image.Image, List[Dict[str, Union[int, str, float, List[float]]]]]:
    model = YOLO('yolov8x.pt')  # Load the model

    # Convert PIL Image to OpenCV image
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert frame to RGB format
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run the model on the image
    results = model.predict(img)

    # Draw bounding boxes and labels on the frame
    annotator = Annotator(frame)
    boxes = []
    for r in results:
        boxes.extend(r.boxes)
        for box in r.boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
    frame = annotator.result()

    # Convert OpenCV image back to PIL Image
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Create a list of results
    output = [{'class': box.cls.item(), 'class_name': model.names[box.cls.item()],
               'xyxy': box.xyxy.tolist()[0], 'conf': box.conf.item()} for box in boxes]

    return image_pil, output




# Function to draw the boxes
def draw_boxes(image: Image.Image, results: List[Dict[str, Union[int, str, float, List[float]]]]) -> Image.Image:
    # Convert PIL Image to OpenCV image
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Loop over the results and draw the bounding boxes
    for result in results:
        x1, y1, x2, y2 = map(int, result['xyxy'])
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Convert OpenCV image back to PIL Image
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    
    return image_pil












