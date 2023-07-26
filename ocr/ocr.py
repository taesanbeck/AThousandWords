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
    image_array = np.array(image_input_rgb)

    prediction_groups = pipeline.recognize([image_array])
    
    # Check if confidence scores are available
    if len(prediction_groups[0][0]) == 3:
        predicted_texts = [(text, box, score) for text, box, score in prediction_groups[0]]  # Also store box coordinates and confidence scores
    else:
        predicted_texts = [(text, box, None) for text, box in prediction_groups[0]]  # Also store box coordinates but no confidence scores

    # If no text found...
    if not predicted_texts:
        return [], image_array  # Return original image if no text found
    else:
        texts = [text for text, _, _ in predicted_texts]
        boxes = [box for _, box, _ in predicted_texts]
        scores = [score for _, _, score in predicted_texts]
        # Iterate over the boxes and draw rectangles on the image
        if bounding_box_option == 'Yes':
            for box, score in zip(boxes, scores):
                # keras-ocr returns boxes as lists of points apparently and We need to convert these to tuples(this was a pain to understand)
                # Also, OpenCV expects coordinates as integers
                box = [(int(x), int(y)) for x, y in box]
                cv2.polylines(image_array, [np.array(box)], isClosed=True, color=(0, 255, 0), thickness=2)
                if score is not None:
                    # Draw confidence score on the image
                    cv2.putText(image_array, f"{score:.2f}", (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    return texts, image_array











  