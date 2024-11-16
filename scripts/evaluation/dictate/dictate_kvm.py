import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def concat_kv(split_kv, num_memory):  
    '''
    This function convert a batched splited memory KV cache into a batched concatenated memory KV cache
    split_kv: ((batch_size * num_memory, num_heads, seq_len, hidden_dims),(...))
    
    final_past_key_values: ((batch_size, num_heads, seq_len * num_memory, hidden_dims),(...))
    '''
    num_layers = len(split_kv)
    split_batch_size = split_kv[0][0].size(0)
    final_past_key_values = ()

    for layer_idx in range(num_layers):
        key_cache, value_cache = split_kv[layer_idx]

        concatenated_keys_list = []
        concatenated_values_list = []

        for i in range(0, split_batch_size, num_memory):
            
            key_group = key_cache[i:i+num_memory]
            key_list = torch.split(key_group, 1, dim=0)
            value_group = value_cache[i:i+num_memory]
            value_list = torch.split(value_group, 1, dim=0)

            concatenated_key = torch.cat(key_list, dim=2)
            concatenated_value = torch.cat(value_list, dim=2)

            concatenated_keys_list.append(concatenated_key)
            concatenated_values_list.append(concatenated_value)

        layer_concatenated_keys = torch.cat(concatenated_keys_list, dim=0)  # Concatenate along batch dimension
        layer_concatenated_values = torch.cat(concatenated_values_list, dim=0)

        final_past_key_values += ((layer_concatenated_keys, layer_concatenated_values),)
        
    return final_past_key_values

def compute_f1_token_ids(generated_sequence, ground_truth_sequence):
    # Flatten the sequences
    gen_tokens = [token for seq in generated_sequence for token in seq]
    gt_tokens = [token for seq in ground_truth_sequence for token in seq]
    
    # Count overlapping tokens (with or without considering duplicates)
    # Option 1: Considering duplicates (counts)
    from collections import Counter
    gen_token_counts = Counter(gen_tokens)
    gt_token_counts = Counter(gt_tokens)
    
    common_tokens = gen_token_counts & gt_token_counts  # Intersection: min counts
    overlap = sum(common_tokens.values())
    
    # Option 2: Ignoring duplicates (set intersection)
    # overlap = len(set(gen_tokens) & set(gt_tokens))
    
    # Compute precision and recall
    precision = overlap / len(gen_tokens) if gen_tokens else 0
    recall = overlap / len(gt_tokens) if gt_tokens else 0
    
    # Compute F1 score
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return f1

def generate_kv_with_id(model, input_ids, p_id):
    input_ids = input_ids.to(model.device)
    p_id = p_id.to(model.device)
    with torch.no_grad():
        out = model(input_ids, position_ids = p_id)
        past_key_values = out.past_key_values

    return past_key_values

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

def load_data(index):

    dataset = load_dataset("openwebtext")

    return dataset['train'][:index]['text']
# Specify the model name; ensure you have access and it is properly installed
# model_name = 'meta-llama/Llama-3.2-1B-Instruct'
model_name = '/mnt/data/jingbo/kv_dump_combine_mix5_30000steps_warmup0.1_decaycosine_5e-6_full/checkpoint-30000'

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

raw_data = load_data(1000)

for i in range(1):
    # Tokenize the current context
    input_ids = tokenizer.encode(raw_data[i], return_tensors='pt')
    mem_num = 10
    mem_len = 100

    if input_ids.size(1) < mem_num * mem_len + 1:
        continue
    
    input_ids = input_ids[:, :mem_num * mem_len + 1]

    split_memory_ids = input_ids[:, 1:].reshape(mem_num, mem_len)
    kv_list = []
    mem_start_position = 1

    split_memory_ids_special = torch.cat([torch.tensor([[128256]] * mem_num).to(split_memory_ids.device), \
                               split_memory_ids, \
                                torch.tensor([[128257]] * mem_num).to(split_memory_ids.device)],dim=1)
    
    position_id = torch.arange(mem_start_position, mem_start_position + mem_num * (mem_len + 2)).unsqueeze(0).reshape(mem_num, mem_len + 2)

    kv_cache = generate_kv_with_id(model, split_memory_ids_special, position_id)

    memory_kv = concat_kv(kv_cache, mem_num)
    begin_kv = generate_kv_with_id(model, torch.tensor([[128000]]), torch.tensor([[0]]))

    past_key_values =  append_kv([begin_kv, memory_kv])

    start_position = 550
    needle_len = 10
    needle = input_ids[:, start_position:start_position + needle_len]

    question_ids = torch.cat([tokenizer.encode("<MEM_SUM>What is the continuation after", return_tensors='pt', add_special_tokens= False), needle, tokenizer.encode("?", return_tensors='pt', add_special_tokens= False)], dim =1)
    print(question_ids.shape)
    concat_ids = torch.cat([torch.tensor([[128000]]), split_memory_ids_special.reshape(1, mem_num * (mem_len + 2)), question_ids], dim = 1)
    print(concat_ids.shape, past_key_values[0][0].shape)
    # Generate new tokens
    output = model.generate(
        concat_ids,
        max_new_tokens=20,        # Generate three tokens each time
        do_sample=False,          # Enable sampling to introduce variability
        temperature=None,
        top_p=1.0,
        past_key_values=past_key_values,
        use_cache=True
    )
    # Extract the newly generated tokens
    new_tokens = output[:, concat_ids.shape[-1]:]
    print(new_tokens)
    ground_truth = input_ids[:, start_position + needle_len : start_position + needle_len + 20]
    print(ground_truth)
    print(compute_f1_token_ids(new_tokens, ground_truth))


    # Decode the new tokens to text
    # print(tokenizer.decode(output[0], skip_special_tokens=False))
    # print(tokenizer.decode(new_tokens[0], skip_special_tokens=False))
    # print(tokenizer.decode(ground_truth[0], skip_special_tokens=False))
    # print(tokenizer.decode(input_ids[0][start_position:start_position + 30], skip_special_tokens=False))



