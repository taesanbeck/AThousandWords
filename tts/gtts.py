# This is for Sukhdeeps Google Text to Speech Function
from gtts import gTTS
from io import BytesIO

def texttospeech(words):
    sound_file = BytesIO()
    tts = gTTS(words, lang='en')
    tts.write_to_fp(sound_file)

texttospeech()

#to process file in streamlit
st.audio(sound_file)