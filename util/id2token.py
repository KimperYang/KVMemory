from transformers import AutoTokenizer
import torch

# Initialize the tokenizer (replace with your model's tokenizer if different)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Example tensor input_ids
ids = torch.tensor([[   1,   518, 25580, 29962]])

print(ids.size(1))
# Convert tensor to list of IDs
id_list = ids.squeeze().tolist()  # Remove any extra dimensions and convert to a list

# Convert IDs to tokens
tokens = tokenizer.convert_ids_to_tokens(id_list)

print("Token list:", tokens)
