from ultralytics import YOLO

def standalone_yolo(image, confidence, save_img=False):
    model=YOLO('yolov8x.pt') # will download the model if it isn't already there
    detection = model(image, conf=confidence, save=save_img, project='objects', name='saved_img')
    output = [{'class': box.cls.item(), 'class_name': detection[0].names[box.cls.item()],
               'xyxy': box.xyxy.tolist()[0], 'conf': box.conf.item()} for box in detection[0].boxes]

    return output

def output_class_list(olist):  # get only the class predictions as human readable names

    return [o['class_name'] for o in olist]

