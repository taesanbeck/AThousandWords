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
    image = np.array(image_input_rgb)  # Convert the image to a numpy array
    image_with_boxes = image.copy()  # Copy the image. We'll draw bounding boxes on this copy

    prediction_groups = pipeline.recognize([image])  # Not the copy with potential boxes

    if not prediction_groups or not prediction_groups[0]:  # Check if prediction_groups is empty or contains an empty group
        return [], image  # Return original image if no text found

    predicted_texts = [word_box[0] for word_box in prediction_groups[0]]  # Extract texts

    if bounding_box_option == 'Yes':
        boxes = [word_box[1] for word_box in prediction_groups[0]]  # Extract boxes only if needed
        # Iterate over the boxes and draw rectangles on the image
        for box in boxes:
            # keras-ocr returns boxes as lists of points apparently and We need to convert these to tuples(this was a pain to understand)
            # Also, OpenCV expects coordinates as integers
            box = [(int(x), int(y)) for x, y in box]
            cv2.polylines(image_with_boxes, [np.array(box)], isClosed=True, color=(0, 255, 0), thickness=2)

        return predicted_texts, image_with_boxes

    else:
        return predicted_texts, image















  