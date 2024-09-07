import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss
import pandas as pd    
import json
import datetime

global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
global_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map="auto")

def generate_kv(prompt):

    tokenizer = global_tokenizer
    model = global_model

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    input_ids = input_ids.to(model.device)
    out = model(input_ids)
    past_key_values = out.past_key_values

    #filter <s>
    filtered_past_key_values = ()

    for past_keys, past_values in past_key_values:

        filtered_keys = past_keys[:, :, 1:, :] 
        filtered_values = past_values[:, :, 1:, :] 
        filtered_past_key_values = filtered_past_key_values + ((filtered_keys, filtered_values),)

    input_ids = input_ids[:, 1:]

    print(filtered_past_key_values[0][0].size())

    return input_ids, filtered_past_key_values

def append_kv(kv_list):
    if not kv_list:
        raise ValueError("kv_list is empty. It must contain at least one past_key_values list.")

    num_layers = len(kv_list[0])

    concatenated_past_key_values = ()

    for layer in range(num_layers):
        
        keys_list = [kv[layer][0] for kv in kv_list]
        values_list = [kv[layer][1] for kv in kv_list]

        concatenated_keys = torch.cat(keys_list, dim=2)
        concatenated_values = torch.cat(values_list, dim=2) 

        concatenated_past_key_values = concatenated_past_key_values + ((concatenated_keys, concatenated_values),)

    return concatenated_past_key_values

def concat_input_id(id_list):

    concat_input_ids = torch.cat([inp['input_ids'][:, 1:] for inp in id_list], dim=1)

    concat_attention_mask = torch.cat([inp['attention_mask'][:, 1:] for inp in id_list], dim=1)

    concatenated_input = {
        'input_ids': concat_input_ids,
        'attention_mask': concat_attention_mask
    }

    return concatenated_input

memory_list = ["<s>", "Hi", "How are you", "I am good"]

id_list = []
kv_list = []

for st in memory_list[:2]:
    _, kv = generate_kv(st)
    kv_list.append(kv)

for id in memory_list:
    id = global_tokenizer(id, return_tensors="pt")
    id_list.append(id)

# inputs = torch.cat(id_list, dim=1)
# inputs = global_tokenizer(memory_list[1], return_tensors="pt")

inputs = concat_input_id(id_list)
past_key_values =  append_kv(kv_list)
print(inputs)
# inputs = global_tokenizer("Hi How are you I am good", return_tensors="pt")

# past_key_values = global_model(global_tokenizer("Hi", return_tensors="pt").input_ids).past_key_values

outputs = global_model(inputs["input_ids"], labels=inputs["input_ids"], past_key_values=past_key_values)
logits = outputs.logits

shift_logits = logits[..., :-1, :].contiguous()
shift_labels = inputs["input_ids"][..., 1:].contiguous()

loss_fct = CrossEntropyLoss(reduction='none')
losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

losses = losses.view(shift_logits.size(0), -1)

mask = torch.ones_like(losses)
mask[:, :2] = 0 

print(mask)

masked_losses = losses * mask
print(masked_losses)
final_loss = masked_losses.sum() / mask.sum()

print(f"Final Loss: {final_loss.item()}")
