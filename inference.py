import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DynamicCache
import json

def generate_kv(prompt):

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map="auto")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model(**inputs, use_cache=True)
    past_key_values = out.past_key_values

    #filter <s>
    filtered_past_key_values = []

    for past_keys, past_values in past_key_values:

        filtered_keys = past_keys[:, :, 1:, :] 
        filtered_values = past_values[:, :, 1:, :] 
        filtered_past_key_values.append((filtered_keys, filtered_values))

    print(filtered_past_key_values[0][0].size())
    return filtered_past_key_values

def append_kv(kv_list):
    if not kv_list:
        raise ValueError("kv_list is empty. It must contain at least one past_key_values list.")

    num_layers = len(kv_list[0])

    concatenated_past_key_values = ()

    for layer in range(num_layers):
        
        keys_list = [kv[layer][0] for kv in kv_list]
        values_list = [kv[layer][1] for kv in kv_list]

        # Concatenate keys and values along the sequence length dimension
        concatenated_keys = torch.cat(keys_list, dim=2)
        concatenated_values = torch.cat(values_list, dim=2) 

        concatenated_past_key_values = concatenated_past_key_values + ((concatenated_keys, concatenated_values),)

    return concatenated_past_key_values

def inference_with_kv(prompt, past_key_values, model_name="meta-llama/Llama-2-7b-chat-hf", max_length=200, num_return_sequences=1):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    # past_key_values = (
    #     (keys.to(model.device), values.to(model.device))
    #     for keys, values in past_key_values
    # )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    # inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)

    model.eval()

    with torch.no_grad():

        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            past_key_values=past_key_values,
            use_cache=True
        )
    
    generated_sequences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return generated_sequences

memory_list = ["hello\n", "how are you\n", "and how old are you\n"]
template = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible.\n<</SYS>>\n"
start_token = "<s>"
end_token = "[/INST]"
memory_list.insert(0, template)
memory_list.insert(0, start_token)
memory_list.append(end_token)

kv_list = []
seq = ""

# for i in range(len(memory_list)):
#     memory_list[i] = memory_list[i]

for st in memory_list:
    kv_list.append(generate_kv(st))
    appended_kv = append_kv(kv_list)
    seq = seq + st

print(appended_kv[0][0].size())
print(seq)

print(inference_with_kv(seq, appended_kv))



# start_kv = generate_kv("<s>")
# template_kv = generate_kv("[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible.\n<</SYS>>\n\n [/INST]")
# dummy_kv1 = generate_kv("hello") 
# dummy_kv2 = generate_kv("how are you")
# dummy_kv3 = generate_kv("and how old are you")


# appended_kv = append_kv([template_kv, start_kv, dummy_kv1, dummy_kv2, dummy_kv3])
# print(appended_kv[0][0].size())
# print(inference_with_kv("[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible.\n<</SYS>>\n\n [/INST] hello ha ha ha how are you and how old are you", appended_kv))
