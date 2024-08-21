import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss
import pandas as pd    
import json
import datetime
from datasets import load_dataset

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

    # print(filtered_past_key_values[0][0].size())

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

def load_data(index):

    dataset = load_dataset("openwebtext")

    return dataset['train'][:index]['text']

def memory_seg(memory_str, seg_method = 'paragraph', chunk_size = 50):

    if seg_method == 'paragraph':

        memory_list = memory_str.split('\n')

        if not len(memory_list) == 1:

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

    num_data_used = 2000
    raw_data = load_data(num_data_used)

    res_dict = {}
    loss_list = []

    chunk_method = 'paragraph' # ['fixsize', 'paragraph']

    for i in range(len(raw_data)):

        print(str(i))

        if(len(raw_data[i].split())>5000): 
            print(str(len(raw_data[i])),"skip!")
            res_dict["Num_Token_"+str(i)] = "skip"
            res_dict["Len_Mask_"+str(i)] = "skip"
            res_dict["Loss_"+str(i)] = "skip"
            continue

        memory_list = memory_seg(raw_data[i], chunk_method)

        start_token = "<s>"
        memory_list.insert(0, start_token)

        id_list = []
        kv_list = []

        for st in memory_list[:len(memory_list)//2]:
            _, kv = generate_kv(st)
            kv_list.append(kv)

        for id in memory_list:
            id = global_tokenizer(id, return_tensors="pt")
            id_list.append(id)

        # inputs = torch.cat(id_list, dim=1)
        # inputs = global_tokenizer(memory_list[1], return_tensors="pt")

        inputs = concat_input_id(id_list)
        past_key_values =  append_kv(kv_list)

        num_token_masked = past_key_values[0][0].size()[2]
        num_token_total = inputs["input_ids"].size(1)
        print("Total Tokens: ", num_token_total)
        print("Masked Tokens: ", num_token_masked)
        # inputs = global_tokenizer("Hi How are you I am good", return_tensors="pt")

        # past_key_values = global_model(global_tokenizer("Hi", return_tensors="pt").input_ids).past_key_values

        outputs = global_model(inputs["input_ids"], labels=inputs["input_ids"], past_key_values=None)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction='none')
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        losses = losses.view(shift_logits.size(0), -1)

        mask = torch.ones_like(losses)
        mask[:, :num_token_masked] = 0 

        masked_losses = losses * mask
        final_loss = masked_losses.sum() / mask.sum()

        print(f"Final Loss: {final_loss.item()}")

        if not str(final_loss.item()) == "nan":
            loss_list.append(final_loss.item())

        res_dict["Num_Token_"+str(i)] = str(num_token_total)
        res_dict["Len_Mask_"+str(i)] = str(num_token_masked)
        res_dict["Loss_"+str(i)] = str(final_loss.item())
    
    res_dict["Average_Final_Loss"] = str(sum(loss_list) / len(loss_list))

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")
    
    file_name = f"result/openwebtext_baseline_{str(num_data_used)}_{chunk_method}_{time_str}.json"

    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(res_dict, file, ensure_ascii=False, indent=4)

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
