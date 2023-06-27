from objects.yolo import standalone_yolo, output_class_list
from PIL import Image
import streamlit as st

@st.cache_resource(ttl="1.5 days", max_entries=10, show_spinner="Loading model...")

def show_page(selected_model):
    st.title('Model Testing')

    st.header('Upload an image:')
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    st.header('Confidence Level:')
    confidence_level = st.slider('Adjust the confidence level', min_value=0.0, max_value=1.0, value=0.5)

    st.header('Bounding Boxes:')
    bounding_box_option = st.radio('Would you like bounding boxes displayed?', ('Yes', 'No'))

    def run_model():
        if uploaded_file is not None:
#        image = Image.open(uploaded_file)  # to make things easier on you, I'm making these functions accept the path
#and do the open, instead of opening it here
             
            if selected_model == 'YOLO':
        # Run the YOLO model on the image

                if bounding_box_option == 'Yes':
                    results, image = standalone_yolo(uploaded_file, confidence=confidence_level, save_img=True,
                                                  save_conf=True)
                if bounding_box_option == 'No':
                    results, image = standalone_yolo(uploaded_file, confidence=confidence_level, save_img=False,
                                                 save_conf=False)

            #now, it will display an image no matter what is chosen
                with Image.open(image) as ifile:
                    st.image(ifile, caption='Uploaded Image', use_column_width=True)  # Display the uploaded image

                labels = output_class_list(results)

            # Display the labels
                st.header('Computer Vision Labels:')
                st.text(labels)

    st.button(label='Run Model', on_click=run_model)
