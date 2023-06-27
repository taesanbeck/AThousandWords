from objects.yolo import standalone_yolo, output_class_list
from PIL import Image
import streamlit as st

#@st.cache_resource(ttl="1.5 days", max_entries=10, show_spinner="Loading model...")
#model=YOLO('https://thousandwordsgmu.s3.amazonaws.com/yolov8x.pt')

# This isn't being invoked anywhere so what is it doing?

def run_model(uploaded_file, selected_model, bounding_box_option, confidence_level):
    if uploaded_file is not None:
        image_input = Image.open(uploaded_file)
        image_name = uploaded_file.name

        if selected_model == 'YOLO':
            # Run the YOLO model on the image

            if bounding_box_option == 'Yes':
                results, image_output = standalone_yolo(image_input, image_name=image_name,
                                                        confidence=confidence_level, save_img=True, save_conf=True)
            if bounding_box_option == 'No':
                results, image_output = standalone_yolo(image_input, image_name=image_name,
                                                        confidence=confidence_level, save_img=False, save_conf=False)

            st.image(image_output, caption='Uploaded Image', use_column_width=True)  # Display the uploaded image

            labels = output_class_list(results)

            # Display the labels
            st.header('Computer Vision Labels:')
            st.text(labels)

def show_page(selected_model):
    st.title('Model Testing')

    st.header('Upload an image:')
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    st.header('Confidence Level:')
    confidence_level = st.slider('Adjust the confidence level', min_value=0.0, max_value=1.0, value=0.5)

    st.header('Bounding Boxes:')
    bounding_box_option = st.radio('Would you like bounding boxes displayed?', ('Yes', 'No'))

    st.button(label="Run Model", on_click=run_model, kwargs={'uploaded_file': uploaded_file, 'confidence_level': confidence_level,
                                                             'bounding_box_option': bounding_box_option, 'selected_model': selected_model})

