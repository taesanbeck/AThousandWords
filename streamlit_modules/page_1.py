from objects.yolo import standalone_yolo, output_class_list, output_class_list_w_meta
from PIL import Image
import streamlit as st
from tts.gtts import texttospeech  
from nlp.t5_coco import generate_caption, model_instance

#@st.cache_resource(ttl="1.5 days", max_entries=10, show_spinner="Loading model...")
#model=YOLO('https://thousandwordsgmu.s3.amazonaws.com/yolov8x.pt')

# This isn't being invoked anywhere so what is it doing?

def run_model(uploaded_file, selected_cv_model, bounding_box_option, confidence_level, selected_nlp_model):
    if uploaded_file is not None:
        image_input = Image.open(uploaded_file)
        image_name = uploaded_file.name

        # It would be a good idea to check and see if predict_filename is already in saved_img
        # if it is, don't run, just show the image.
        # however, we would also need to save the results dict, possibly as json, so we can call it back up.

        if selected_cv_model == 'YOLO':
            # Run the YOLO model on the image

            if bounding_box_option == 'Yes':
                results, image_output = standalone_yolo(image_input, image_name=image_name,
                                                        confidence=confidence_level, save_img=True)
            if bounding_box_option == 'No':
                results, image_output = standalone_yolo(image_input, image_name=image_name,
                                                        confidence=confidence_level, save_img=False)

            st.image(image_output, caption='Uploaded Image', use_column_width=True)  # Display the uploaded image

            # watch this
            labels = output_class_list_w_meta(results)
            
            # Display the labels
            st.header('Computer Vision Labels:')
            st.text(str(labels))
            
        elif selected_cv_model == 'DINO':
        # Add code here to run the DINO model on the image
            pass

        # Generate a sentence from the labels using the selected NLP model
        if selected_nlp_model == 'T5':
            caption = generate_caption(model_instance, labels)
            
            #Filter out the uneeded output from yolo/T5
            import re
            caption = re.sub(r'\d+ ', '', caption.replace(':', '').replace('@', 'at').replace('a caption',''))

            
            # Display the generated sentence
            st.title('Generated Caption:')
            st.text(caption)
 
            # Generate audio file for caption and play it
            texttospeech(caption)  # Convert caption to audio
            audio_file = open("output.mp3", "rb")
            st.audio(audio_file.read(), format='audio/mp3')  # Play audio
            audio_file.close()

            
        elif selected_nlp_model == 'GPT2':
            # Add code here to generate a caption using GPT2
                pass


def show_page(selected_cv_model, selected_nlp_model):
    st.title('Model Testing')

    st.header('Upload an image:')
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    st.header('Confidence Level:')
    confidence_level = st.slider('Adjust the confidence level', min_value=0.0, max_value=1.0, value=0.5)

    st.header('Bounding Boxes:')
    bounding_box_option = st.radio('Would you like bounding boxes displayed?', ('Yes', 'No'))

    st.button(label="Run Model", on_click=run_model, kwargs={'uploaded_file': uploaded_file,
                                                            'confidence_level': confidence_level,
                                                            'bounding_box_option': bounding_box_option,
                                                            'selected_cv_model': selected_cv_model,
                                                            'selected_nlp_model': selected_nlp_model})
    
    

