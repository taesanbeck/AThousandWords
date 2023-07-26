# ocr.py this is the run ocr function for detecting text
import cv2
import numpy as np
import keras_ocr
from PIL import Image
import io

# Get a set of three pre-trained models
pipeline = keras_ocr.pipeline.Pipeline()

def run_ocr(image_data, bounding_box_option):
    # Convert BytesIO to PIL Image
    image_input = Image.open(io.BytesIO(image_data))

    # Convert PIL Image to RGB and then to numpy array for OpenCV
    image_input_rgb = image_input.convert("RGB")
    image_with_boxes = np.array(image_input_rgb)  # Renamed variable to match show_page function

    prediction_groups = pipeline.recognize([image_with_boxes])
    predicted_texts = [(text, box) for text, box in prediction_groups[0]]  # Also store box coordinates

    # If no text found...
    if not predicted_texts:
        return [], image_with_boxes  # Return original image if no text found
    else:
        texts = [text for text, _ in predicted_texts]  # Separate texts into a different variable
        boxes = [box for _, box in predicted_texts]  # Boxes in separate variable
        # Iterate over the boxes and draw rectangles on the image only if bounding_box_option is 'Yes'
        if bounding_box_option == 'Yes':
            for box in boxes:
                # keras-ocr returns boxes as lists of points apparently and We need to convert these to tuples(this was a pain to understand)
                # Also, OpenCV expects coordinates as integers
                box = [(int(x), int(y)) for x, y in box]
                cv2.polylines(image_with_boxes, [np.array(box)], isClosed=True, color=(0, 255, 0), thickness=2)

    return texts, image_with_boxes  # Return texts and image_with_boxes


















  