import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify the model name; ensure you have access and it is properly installed
# model_name = 'meta-llama/Llama-3.2-1B-Instruct'
model_name = '/mnt/data/jingbo/kv_dump_combine_mix5_30000steps_warmup0.1_decaycosine_5e-6_full/checkpoint-30000'

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initial prompt
prompt = "Today is really a sunny day. I want to go for a walk. What is the continuation after Today is really?"

# Set the initial context
context = prompt

# Number of iterations you want the loop to run
num_iterations = 1

for _ in range(num_iterations):
    # Tokenize the current context
    input_ids = tokenizer.encode(context, return_tensors='pt')

    # Generate new tokens
    output = model.generate(
        input_ids,
        max_new_tokens=10,        # Generate three tokens each time
        do_sample=False,          # Enable sampling to introduce variability
        # top_k=50,                # Consider the top 50 tokens by probability
        # top_p=0.95,              # Or cumulative probability of 95%
        # temperature=0.7,         # Control the randomness
        # num_return_sequences=1   # Generate one sequence
    )

    # Extract the newly generated tokens
    new_tokens = output[0][input_ids.shape[-1]:]

    # Decode the new tokens to text
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Append the new text to the context
    context += new_text

    # Print the newly generated text
    print(new_text)
