# t5_coco.py Module For Streamlit
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn as nn
import sentencepiece



class New_T5_Trainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def generate(self, input_ids, decoder_input_ids=None, **kwargs):
        return self.model.generate(input_ids, decoder_input_ids=decoder_input_ids, **kwargs)

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = New_T5_Trainer()
model.load_state_dict(torch.load('nlp/t5_coco.pt'))  # load the weights

def generate_caption(model, input_keywords):
    # Prepare the input data
    input_text = 'generate caption: ' + ' '.join(input_keywords)
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Detect the device of the model
    device = next(model.parameters()).device

    # Move the input tensors to the same device as the model
    input_ids = input_ids.to(device)

    # Generate output from the model
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=80,
            min_length=5,
            num_beams=5,
            temperature=0.85,
            no_repeat_ngram_size=4,
            do_sample=True,
            top_k=2,
            early_stopping=True,
        )

    # Decode the output tokens to text
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the 'A caption: ' part from the output text
    output_text = output_text.replace('A caption: ', '')

    return output_text


    # Instantiate the model
model_instance = New_T5_Trainer()

# Load the model weights from the saved file
model_instance.load_state_dict(torch.load('/home/ec2-user/AThousandWords/nlp/t5_coco.pt'))


# Move the model to GPU if available
# model_instance.to('cuda')

# EXAMPLE for how to use loaded model to generate captions
input_keywords = ['Zebras', 'seen', 'eating', 'hay', 'large', 'stall']
caption = generate_caption(model_instance, input_keywords)
print("Generated Caption:", caption)

######### RUN THE MODEL
import streamlit as st
from tts.texttospeech import texttospeech

def run_t5(labels):
    caption = generate_caption(model_instance, labels)

    # Display the generated sentence
    st.title('Generated Caption:')
    st.text(caption)

    # Display the labels
    st.subheader('Computer Vision Labels:')
    st.text(labels)

    # Generate audio file for caption and play it
    texttospeech(caption)  # Convert caption to audio
    audio_file = open("output.mp3", "rb")
    st.audio(audio_file.read(), format='audio/mp3')  # Play audio
    audio_file.close()
