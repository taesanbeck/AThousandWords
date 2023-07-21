from imageai.Detection import ObjectDetection
import os
from PIL import Image
import streamlit as st

def run_yolo3(image_input, image_name, confidence_level, bounding_box_option):
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path , "yolov3.pt"))
    detector.loadModel()
    
    # Convert the input image to an RGB image
    image_input = image_input.convert('RGB')
    
    # Specify the path to the output image
    output_image_path = os.path.join(execution_path, "results", "imagenew.jpg")
    
    # Create the results directory if it does not already exist
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    
    detections = detector.detectObjectsFromImage(input_image=image_input,
                                                 output_image_path=output_image_path,
                                                 minimum_percentage_probability=confidence_level)
    labels = []
    for eachObject in detections:
        labels.append(eachObject["name"])
    
    # Read and resize the output image
    output_image = Image.open(output_image_path)
    output_image = output_image.resize((608, 608))
    
    # Display the output image only if bounding_box_option is 'Yes'
    if bounding_box_option == 'Yes':
        st.image(output_image, caption='Output Image', use_column_width=True)
    else:  # Add this block
        # Display the original image without bounding boxes
        st.image(image_input, caption='Uploaded Image', use_column_width=True)
    
    # Delete the output image and its parent directory
    os.remove(output_image_path)
    os.rmdir(os.path.dirname(output_image_path))
    
    labels = ' '.join(labels)
 
    return labels






    

