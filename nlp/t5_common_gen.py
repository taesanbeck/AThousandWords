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

def run_t5_common_gen(labels):
    model_instance = Run_T5_Common_Gen()
    input_text = ' '.join(labels)
    generated_text = model_instance.generate_text(input_text)

    # Display the generated text
    st.title('Generated Text:')
    st.text(generated_text)

    # Display the input text
    st.subheader('Input Text:')
    st.text(input_text)

    # Generate audio file for generated text and play it
    texttospeech(generated_text)  # Convert generated text to audio
    audio_file = open("output.mp3", "rb")
    st.audio(audio_file.read(), format='audio/mp3')  # Play audio
    audio_file.close()

