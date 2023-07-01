# Google Text to Speech Function
from gtts import gTTS
from io import BytesIO

def texttospeech(words):
    sound_file = BytesIO()
    tts = gTTS(words, lang='en')
    tts.save("output.mp3")
