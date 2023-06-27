from ultralytics import YOLO
import os

def standalone_yolo(image, confidence, save_img, save_conf):
    image_filename = os.path.split(image)[-1]
    model=YOLO('https://thousandwordsgmu.s3.amazonaws.com/yolov8x.pt') # will download the model if it isn't already there
    detection = model(image, conf=confidence, save=save_img, project='objects', name='saved_img', save_conf=save_conf)
    output = [{'class': box.cls.item(), 'class_name': detection[0].names[box.cls.item()],
               'xyxy': box.xyxy.tolist()[0], 'conf': box.conf.item()} for box in detection[0].boxes]
    if save_img == False:
        image_path = detection[0].path # if you want this to be the Image object, just change it to image_path = image
    if save_img == True:
        image_path = os.path.join('objects', 'saved_img', image_filename)

    return output, image_path

def output_class_list(olist):  # get only the class predictions as human readable names

    return [o['class_name'] for o in olist]


# if you need to make it show the image, you can just do results[0].plot() instead of importing a new package
# and writing an entirely new function










