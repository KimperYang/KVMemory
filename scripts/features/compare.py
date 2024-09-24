import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.models.llama.modeling_llama
# Load the model and tokenizer
model_name = 'meta-llama/Llama-2-7b-hf'  # replace with your model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="auto")
# model.to('cuda')
# print(model.__class__)
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
def get_kv_cache(input_ids, position_ids=None, mask=None):
    with torch.no_grad():
        outputs = model(input_ids, position_ids=position_ids, attention_mask=mask, use_cache=True)
    # print(outputs.past_key_values[0][0].size())
    return outputs.past_key_values

input_ids1 = torch.tensor([[  306,  5360,  4796, 11203]])
input_ids2 = torch.tensor([[  0 , 0, 0, 0, 306,  5360,  4796, 11203]])
# Get the KV caches with different position ids
kv_cache_1 = get_kv_cache(input_ids)

kv_cache_2 = get_kv_cache(input_ids2)
kv_cache_3 = get_kv_cache(input_ids, position_ids_3)
# for i in range(len(kv_cache_1)):
#     print(i)
#     print(kv_cache_1[i][1][0][16][1][7])
#     print(kv_cache_3[i][1][0][16][1][7])
def slice_kv(kv):
    new_kv = ()
    for i in range(len(kv)):
        k, v = kv[i]
        k = k[:, :, 4:, :]
        v = v[:, :, 4:, :]
        new_kv += ((k,v),)
    return new_kv

# # Compare the KV caches
def compare_kv_caches(kv_cache_a, kv_cache_b):
    key_diff = 0
    value_diff = 0
    # for layer in range(len(kv_cache_a)):
    key_diff += torch.sum(torch.abs(kv_cache_a[1][0] - kv_cache_b[1][0]))
    value_diff += torch.sum(torch.abs(kv_cache_a[1][1] - kv_cache_b[1][1]))
    return key_diff.item(), value_diff.item()

kv_cache_2 = slice_kv(kv_cache_2)
# print(kv_cache_2[0][0].shape)

key_diff_1_2, value_diff_1_2 = compare_kv_caches(kv_cache_1, kv_cache_2)
key_diff_1_3, value_diff_1_3 = compare_kv_caches(kv_cache_1, kv_cache_3)

print(f"Key difference between position IDs [0,n] and [n,2n]: {key_diff_1_2}")
print(f"Value difference between position IDs [0,n] and [n,2n]: {value_diff_1_2}")

print(f"Key difference between position IDs [0,n] and [2n,3n]: {key_diff_1_3}")
print(f"Value difference between position IDs [0,n] and [2n,3n]: {value_diff_1_3}")

print(torch.mean(kv_cache_1[1][1]))
print(torch.mean(kv_cache_2[1][1]))

print(kv_cache_1[1][1][0][7][2])
print(kv_cache_2[1][1][0][7][2])
# # print('/////////////////')
# # print(kv_cache_1[10][1])
# # print('/////////////////')
# # print(kv_cache_2[10][1])
# # print('/////////////////')
# # print(kv_cache_3[10][1])