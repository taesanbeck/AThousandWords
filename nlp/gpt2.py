# GPT2 function
# Pass in the object labels variable and it should spit out some text

from transformers import GPT2Tokenizer, GPT2LMHeadModel

def describe_image(labels):
    # Initialize GPT2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # If 'pad_token' is not defined, set it as 'eos_token'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Join labels with commas and turn into a string
    label_string = ", ".join(labels)
    # Add a prompt for the GPT2 model to generate a description from
    prompt = f"Describe a scene where a {label_string} are interacting with each other."
    # Encode the prompt and create attention_mask
    inputs = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True)
    attention_mask = inputs.ne(tokenizer.pad_token_id).float()
    # Generate a text from the prompt. Adjust temperature up for more randomness, and max_length for longer outputs. top_p is top probability threshhold for words. ex: 0.5 would only consider the smallest possible set of words whos cumulative probability exceeds 0.5
    outputs = model.generate(inputs, max_length=80, temperature=0.4, do_sample=True, top_p=0.4, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id)
    # Decode the output
    output_text = tokenizer.decode(outputs[0])
    return output_text