import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_name = 'meta-llama/Llama-2-7b-hf'  # replace with your model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
model.eval()

# The sentence to test
sentence = "I love white dog"
input_ids = tokenizer.encode(sentence, return_tensors='pt').to(model.device)
input_ids = input_ids[:,1:]
print(input_ids)

# Function to generate position ids with different ranges
def generate_position_ids(n, factor):
    return torch.tensor([list(range(factor * n, factor * n + n))], device=model.device)

# Number of tokens in the sentence
n = input_ids.shape[1]

# Different position id schemes
position_ids_1 = generate_position_ids(n, 0)  # Position IDs: [0, 1, 2, 3]
position_ids_2 = generate_position_ids(n, 1)  # Position IDs: [n, n+1, n+2, n+3]
position_ids_3 = generate_position_ids(n, 2)  # Position IDs: [2n, 2n+1, 2n+2, 2n+3]

print(position_ids_1, position_ids_2, position_ids_3)
# Function to get KV cache
def get_kv_cache(input_ids, position_ids):
    with torch.no_grad():
        outputs = model(input_ids, position_ids=position_ids, use_cache=True)
    # print(outputs.past_key_values[0][0].size())
    return outputs.past_key_values

# Get the KV caches with different position ids
kv_cache_1 = get_kv_cache(input_ids, position_ids_1)
kv_cache_2 = get_kv_cache(input_ids, position_ids_2)
kv_cache_3 = get_kv_cache(input_ids, position_ids_3)

# Compare the KV caches
def compare_kv_caches(kv_cache_a, kv_cache_b):
    key_diff = 0
    value_diff = 0
    for layer in range(len(kv_cache_a)):
        key_diff += torch.sum(torch.abs(kv_cache_a[layer][0] - kv_cache_b[layer][0]))
        value_diff += torch.sum(torch.abs(kv_cache_a[layer][1] - kv_cache_b[layer][1]))
    return key_diff.item(), value_diff.item()

key_diff_1_2, value_diff_1_2 = compare_kv_caches(kv_cache_1, kv_cache_2)
key_diff_1_3, value_diff_1_3 = compare_kv_caches(kv_cache_1, kv_cache_3)

print(f"Key difference between position IDs [0,n] and [n,2n]: {key_diff_1_2}")
print(f"Value difference between position IDs [0,n] and [n,2n]: {value_diff_1_2}")

print(f"Key difference between position IDs [0,n] and [2n,3n]: {key_diff_1_3}")
print(f"Value difference between position IDs [0,n] and [2n,3n]: {value_diff_1_3}")
