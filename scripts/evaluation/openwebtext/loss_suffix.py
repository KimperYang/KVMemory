import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss
import pandas as pd    
import json
import datetime
from datasets import load_dataset

global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
global_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, device_map="auto")

def generate_kv_with_id(input_ids, p_id):
    model = global_model
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        out = model(input_ids, position_ids = p_id)
        past_key_values = out.past_key_values

    #filter <s>
    # filtered_past_key_values = ()

    # for past_keys, past_values in past_key_values:

    #     filtered_keys = past_keys[:, :, 1:, :] 
    #     filtered_values = past_values[:, :, 1:, :] 
    #     filtered_past_key_values = filtered_past_key_values + ((filtered_keys, filtered_values),)

    # print(past_key_values[0][0].size())

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
        # print(concatenated_past_key_values[0][0].size())

    return concatenated_past_key_values

def concat_input_id(id_list):

    concat_input_ids = torch.cat([inp['input_ids'][:, 1:] for inp in id_list], dim=1)

    concat_attention_mask = torch.cat([inp['attention_mask'][:, 1:] for inp in id_list], dim=1)

    concatenated_input = {
        'input_ids': concat_input_ids,
        'attention_mask': concat_attention_mask
    }

    return concatenated_input

def load_data(index):

    dataset = load_dataset("openwebtext")

    return dataset['train'][:index]['text']

def memory_seg(memory_str, seg_method = 'paragraph', chunk_size = 25):

    if seg_method == 'paragraph':

        memory_list = memory_str.split('\n')

        if not len(memory_list) < 4:

            clean_list = []
            for mem in memory_list:
                if not mem == "":
                    clean_list.append(mem)

            return clean_list
        
        else:
            memory_list = memory_str.split('.')

            clean_list = []
            for mem in memory_list:
                if not mem == "":
                    clean_list.append(mem)
            return clean_list
    
    if seg_method == 'fixsize':

        words = memory_str.split()
        memory_list = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

        return memory_list

def main():

    num_data_used = 5000
    raw_data = load_data(num_data_used)

    res_dict = {}
    res_dict["value"] = []
    res_dict["avg"] = 0
    res_dict["num_samples"] = 0
    loss_list = []

    chunk_method = 'fixsize' # ['fixsize', 'paragraph']

    # for i in range(27,28):
    for i in range(len(raw_data)):
        print("Sample No."+ str(i+1))
        tokenized = global_tokenizer(raw_data[i], return_tensors="pt")
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        num_tokens = input_ids.shape[1]

        if num_tokens < 1000:
            continue

        input_ids = input_ids[:, :1000]
        attention_mask = attention_mask[:, 505:1000]

        remaining_ids = input_ids[:, 505:]

        outputs = global_model(input_ids=remaining_ids)

        logits = outputs.logits  

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = remaining_ids[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction='none')
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        losses = losses.view(shift_logits.size(0), -1)

        # print(attention_mask)
        mask = attention_mask[0, 1:].clone()
        # print(mask.size())

        masked_losses = losses * mask
        final_loss = masked_losses.sum() / mask.sum()
        
        print("Loss:", final_loss.item())

        res_dict["value"].append(final_loss.item())

    res_dict["avg"]= sum(res_dict["value"]) / len(res_dict["value"])
    res_dict["num_samples"] = len(res_dict["value"])

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")
    
    file_name = f"result/openwebtext_{str(num_data_used)}_suffix_{time_str}.json"

    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(res_dict, file, ensure_ascii=False, indent=4)

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
