# t5_common_gen.py Module For Streamlit
from happytransformer import HappyTextToText
from happytransformer import TTSettings
from tts.texttospeech import texttospeech
import streamlit as st

class Run_T5_Common_Gen:
    def __init__(self):
        self.happy_common_gen = HappyTextToText("T5", "mrm8488/t5-base-finetuned-common_gen")

    def generate_text(self, input_text, beam_args=TTSettings(num_beams=5, min_length=1, max_length=100)):
        result = self.happy_common_gen.generate_text(input_text, args=beam_args)
        return result.text

model_instance = Run_T5_Common_Gen()  # Instantiate outside the function

def run_t5_common_gen(preprocessed_input):
    # Convert list of labels to a string
    input_string = ' '.join(preprocessed_input)

    generated_text = model_instance.generate_text(input_string)

    # Display the generated text
    st.title('Generated Text:')
    st.text(generated_text)

    # Display the input text
    st.header('Pre-Processed Computer Vision Labels:')
    st.text(input_string)

    return generated_text

