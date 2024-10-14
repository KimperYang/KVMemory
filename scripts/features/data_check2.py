import os
import torch
# import wandb
from torch.optim import AdamW
from transformers import TrainingArguments
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from accelerate import Accelerator
import json

data2 = load_dataset("nvidia/Daring-Anteater")
# data2 = data2['train']['conversations'][:4100]

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
max_length = 2048
def process_conversation(conversation):
        # Extract "Assistant" responses and mask "User" queries
        system = "[INST] <<SYS>>\nYou're an assistant who answer the question with the knowledge provided in the prompt\n<</SYS>>\n\n"
        text = system
        system_tokenized = tokenizer(text, return_tensors="pt")
        system_input_ids = system_tokenized.input_ids
        system_attention_msk = system_tokenized.attention_mask

        mask = []
        mask.extend([0] * system_input_ids.size(1))
        input_ids_list = [system_input_ids]
        attention_mask_list = [system_attention_msk]

        for i in range(len(conversation)):
            
            if conversation[i]["from"] == "User":
                if i==0:
                    t = conversation[i]["value"] + " [/INST] "
                else:
                    t = "<s> [INST] " + conversation[i]["value"]  + " [/INST] " 
                
                tokenized = tokenizer(t, return_tensors="pt")

                input_ids = tokenized.input_ids[:, 1:]
                if len(mask) + input_ids.size(1) > max_length: 
                    break
                attention_msk = tokenized.attention_mask[:, 1:]

                mask.extend([0] * input_ids.size(1))
                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_msk)

            elif conversation[i]["from"] == "Assistant":
                t = conversation[i]["value"]
                
                tokenized = tokenizer(t, return_tensors="pt")

                input_ids = tokenized.input_ids[:, 1:]
                attention_msk = tokenized.attention_mask[:, 1:]
                if len(mask) + input_ids.size(1) > max_length - 1: 
                    input_ids = input_ids[:, :max_length - 1 - len(mask)]
                    attention_msk = attention_msk[:, :max_length - 1 - len(mask)]

                # add </s>
                input_ids = torch.cat([input_ids, torch.tensor([[2]]).to(input_ids.device)], dim = 1)
                attention_msk = torch.cat([input_ids, torch.tensor([[1]]).to(input_ids.device)], dim = 1)

                mask.extend([1] * input_ids.size(1))
                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_msk)
        
        input_ids = torch.cat(input_ids_list, dim=1)
        attention_mask = torch.cat(attention_mask_list, dim=1)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'dataset_id': 'sft',
            'loss_mask': torch.tensor(mask).unsqueeze(0)  # Mask to ignore User's tokens during loss computation
        }

filtered_strings = []

# for i, string in enumerate(data2):
#     print(i)
def validornot(sample):

    mask = process_conversation(sample['conversations'])['loss_mask']
    all_zeros = (mask == 0).all()
    # print(all_zeros.item())
    if all_zeros.item(): 
        return False
        
    else:
        return True

filtered_dataset = data2['train'].filter(validornot)

# Save the filtered dataset to the specified path
filtered_dataset.save_to_disk("/mnt/data2/jingbo/kvmemory/filtered_strings_sft_new")
# print(f"Dataset saved to {save_path}")
      
# output_file = "/mnt/data2/jingbo/kvmemory/filtered_strings_sft.json"
# with open(output_file, "w") as f:
#     json.dump(filtered_strings, f, indent=4)

# print(len(filtered_strings))