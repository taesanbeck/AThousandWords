from ultralytics import YOLO
from PIL import Image
import os
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def standalone_yolo(image, confidence, save_img, image_name):
    model=YOLO('yolov8x.pt') # will download the model if it isn't already there
    detection = model.predict(image, conf=confidence)
    output = [{'class': box.cls.item(), 'class_name': detection[0].names[box.cls.item()],
               'xyxy': box.xyxy.tolist()[0], 'conf': box.conf.item()} for box in detection[0].boxes]
    if save_img == False:
        result_image = image
    if save_img == True:
        if not os.path.exists(os.path.join('objects', 'saved_img')):
            os.mkdir('objects')
            os.mkdir(os.path.join('objects', 'saved_img'))
        raw_output = detection[0].plot(pil=True)
        cv2.imwrite(os.path.join('objects', 'saved_img', 'predict_'+image_name), raw_output)
        result_image = Image.open(os.path.join('objects', 'saved_img', 'predict_'+image_name))

    return output, result_image

def output_class_list(olist):  # get only the class predictions as human readable names

    return [o['class_name'] for o in olist]











