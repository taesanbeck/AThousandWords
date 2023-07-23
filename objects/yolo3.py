from imageai.Detection import ObjectDetection
import os
from PIL import Image
import streamlit as st

def run_yolo3(image_input, image_name, confidence_level, bounding_box_option):
    execution_path = "/home/taesanbeck/AThousandWords/AThousandWords"
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path , "yolov3.pt"))
    detector.loadModel()
    
    # Convert the input image to an RGB image
    image_input = image_input.convert('RGB')
    
    # Save the converted image
    image_input_path = os.path.join(execution_path, image_name)
    image_input.save(image_input_path)

    # Check if the results directory exists and create it if not
    results_dir = os.path.join(execution_path, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Specify the path to the output image
    output_image_path = os.path.join(results_dir, "imagenew.jpg")
    
    detections = detector.detectObjectsFromImage(input_image=image_input_path,  # use the saved image's path here
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
    else: 
        # Display the original image without bounding boxes
        st.image(image_input, caption='Uploaded Image', use_column_width=True)
    
    # Delete the input and output images
    os.remove(image_input_path)
    os.remove(output_image_path)
    
    labels = ' '.join(labels)
 
    return labels
