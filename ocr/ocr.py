# ocr.py this is the run ocr function for detecting text
import cv2
import numpy as np
import keras_ocr
from PIL import Image
import io
import math
import streamlit as st

# Get a set of three pre-trained models
pipeline = keras_ocr.pipeline.Pipeline()

def rotate(box):
  
  tl = box[0][0]
  tr = box[0][1]
  z = (tl[1]-tr[1])/(tr[0]-tl[0])
  rad = math.atan(z)
  s = math.sin(rad)
  c = math.cos(rad)
  boxt = []

  for wrds in box:
    temp = []
    for vert in wrds:
      x = (vert[0]*c) - (vert[1]*s)
      y = (vert[1]*c) + (vert[0]*s)
      temp.append((x,y))
    boxt.append(np.array(temp,dtype = 'float32'))

  return boxt

def order_of_words(box):
    if box[0][0][1] != box[0][1][1]:
       box = rotate(box)

    tly = [-1*point[0][1] for point in box]
    bry = [-1*point[2][1] for point in box]
    tlx = [point[0][0] for point in box]
    words = [item for item in range(0, len(box))]
    runs = 0

    ordered = []
    #some kind of loop must happen here#
    while bool(words):
      temp = []
      wordsy = [tly[h] for h in words]
      maxy = words[wordsy.index(max(wordsy))]

      for z in words:
        if tly[z] > bry[maxy]: #if top y value of a word is higher than bottom y of the highest word, then they're added to the same line. Includes itself 
          temp.append(z)
      xtemp = [tlx[i] for i in temp] #create list of x values for the words in the line
      xsort = sorted(xtemp) #sort the x values for the words in a line in ascending order
      for k in xsort: #stepping through the x values, add the index of the corresponding word to the "ordered" list and remove from the words list
        ordered.append(temp[xtemp.index(k)])
        words.remove(temp[xtemp.index(k)])
      runs = runs+1
      if runs > len(tly): #safeguard to ensure while loop doesn't go on forever in case of an error in code
        break

    return ordered

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
        order = order_of_words(boxes)
        text_order = [texts[i] for i in order]

    return text_order, image_with_boxes  # Return texts and image_with_boxes