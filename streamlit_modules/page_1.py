from objects.yolo import standalone_yolo2, draw_boxes, output_class_list
from PIL import Image
import streamlit as st
from ultralytics import YOLO

@st.cache_resource(ttl="1.5 days", max_entries=10, show_spinner="Loading model...")
def standalone_yolo():
    # Load and return YOLO model
    model = YOLO('yolov8x.pt')
    return model

def show_page(selected_model):
    st.title('Model Testing')

    st.header('Upload an image:')
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    st.header('Confidence Level:')
    confidence_level = st.slider('Adjust the confidence level', min_value=0.0, max_value=1.0, value=0.5)

    st.header('Bounding Boxes:')
    bounding_box_option = st.radio('Would you like bounding boxes displayed?', ('Yes', 'No'))

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
             
        if selected_model == 'YOLO':
        # Run the YOLO model on the image
            image, results = standalone_yolo2(image, confidence=confidence_level)
            
            # Draw bounding boxes on the image
            if bounding_box_option == 'Yes':
                image = draw_boxes(image, results)

            st.image(image, caption='Uploaded Image', use_column_width=True)  # Display the uploaded image

            labels = output_class_list(results)

            # Display the labels
            st.header('Computer Vision Labels:')
            st.text(labels)
