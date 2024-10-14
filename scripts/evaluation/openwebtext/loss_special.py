import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss
import pandas as pd    
import json
import datetime
from datasets import load_dataset
from peft import PeftModel, PeftConfig
import pdb
global_tokenizer = AutoTokenizer.from_pretrained("/mnt/data/jingbo/kv_dump_combine_special2")

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16)
lm_head = base_model.state_dict()['lm_head.weight'].detach().numpy()
vocab_size = len(global_tokenizer)
base_model.resize_token_embeddings(vocab_size)
pdb.set_trace()

peft_config_path = "/mnt/data/jingbo/kv_dump_combine_special2"  # Path to the directory where LoRA weights are stored
lora_config = PeftConfig.from_pretrained(peft_config_path)
test_model = base_model.add_adapter(lora_config)
test_model.enable_adapters()
global_model = PeftModel.from_pretrained(base_model, peft_config_path)

lora_lm_head = global_model.state_dict()['base_model.model.lm_head.modules_to_save.default.weight'].detach().numpy()
original_lm_head = global_model.state_dict()['base_model.model.lm_head.original_module.weight'].detach().numpy()
pdb.set_trace()
global_model.to("cuda")

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

    res_dict = {key: [] for key in range(1, 11)}
    res_dict["avg"] = {key: [] for key in range(1, 11)}
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
        attention_mask = attention_mask[:, :1000].to(global_model.device)

        memory_ids = input_ids[:, 1:505] #filter <s> when calculating kv

        num_tokens_per_part = [504, 252, 168, 126, 101, 84, 72, 63, 56, 51]

        for j in range(len(num_tokens_per_part)):
            split_input_ids = torch.split(memory_ids, num_tokens_per_part[j], dim=1)
            # print(j)

            kv_list = [generate_kv_with_id(global_tokenizer("", return_tensors="pt").input_ids, torch.tensor([[0]]).to(global_model.device))] #initialize with kv cache for <s>

            current_position = 1
            for k in range(len(split_input_ids)):
                tem_input_ids = torch.cat([split_input_ids[k], torch.tensor([[32000]]).to(split_input_ids[k].device)],dim=1)
                position_id = torch.arange(current_position, current_position + tem_input_ids.size(1)).unsqueeze(0)
                kv_cache = generate_kv_with_id(tem_input_ids, position_id.to(global_model.device))
                kv_list.append(kv_cache)
                current_position += tem_input_ids.size(1)

            past_key_values =  append_kv(kv_list)
            remaining_ids = input_ids[:, 505:].to(global_model.device)

            # print(past_key_values[0][0].device, remaining_ids.device, attention_mask.device)

            outputs = global_model(input_ids=remaining_ids, past_key_values=past_key_values, use_cache=True)

            logits = outputs.logits  

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = remaining_ids[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            losses = losses.view(shift_logits.size(0), -1)

            # print(attention_mask)
            mask = attention_mask[0, 1:].clone()
            # print(mask.size())
            mask = mask[505:]

            masked_losses = losses * mask
            final_loss = masked_losses.sum() / mask.sum()
            
            print("Num of Mem:", j+1, " ,Loss:", final_loss.item())

            res_dict[j+1].append(final_loss.item())

    for i in range(1,11):
        res_dict["avg"][i] = sum(res_dict[i]) / len(res_dict[i])
    res_dict["num_samples"] = len(res_dict[1])

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")
    
    file_name = f"result/openwebtext_{str(num_data_used)}_special_{time_str}.json"

    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(res_dict, file, ensure_ascii=False, indent=4)

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
