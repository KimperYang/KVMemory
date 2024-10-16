import os
import torch
# import wandb
from torch.optim import AdamW
from transformers import TrainingArguments
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator
import json

print(len(load_from_disk("/mnt/data2/jingbo/kvmemory/data/maxlen4096/text_min2048")))
# data2 = load_dataset("nvidia/Daring-Anteater")
# # data2 = data2['train']['conversations'][:4100]

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
# max_length = 4096

# def process_conversation(conversation):
#     # Extract "Assistant" responses and mask "User" queries
#     system = "[<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
#     system_tokenized = tokenizer(system, add_special_tokens=False)
#     system_input_ids = system_tokenized.input_ids

#     input_ids_list = system_input_ids
#     labels = [-100] * len(system_input_ids)

#     for i in range(len(conversation)):
        
#         if conversation[i]["from"] == "User":
#             if i==0:
#                 t = conversation[i]["value"] + "<|eot_id|>"
#             else:
#                 t = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[i]["value"]  + "<|eot_id|>" 
#                 # t = " </s><s>[INST] " + conversation[i]["value"]  + " [/INST] " 
            
#             tokenized = tokenizer(t)

#             input_ids = tokenized.input_ids[1:]
#             if len(labels) + len(input_ids) >= max_length: 
#                 break

#             labels.extend([-100] * len(input_ids))
#             input_ids_list += input_ids

#         elif conversation[i]["from"] == "Assistant":
#             t = "<|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[i]["value"]
            
#             tokenized = tokenizer(t)

#             input_ids = tokenized.input_ids[1:]
#             if len(labels) + len(input_ids) > max_length - 1: 
#                 input_ids = input_ids[:max_length - 1 - len(labels)]

#             # # add </s>
#             # input_ids = torch.cat([input_ids, self.eos_token.to(input_ids.device)], dim = 1)
#             # attention_msk = torch.cat([attention_msk, self.mask_token.to(attention_msk.device)], dim = 1) 
#             input_ids += [128009]

#             # if len(mask) + input_ids.size(1) > max_length: 
#             #     input_ids = input_ids[:, :max_length - len(mask)]
#             #     attention_msk = attention_msk[:, :max_length - len(mask)]


#             labels.extend(input_ids)
            
#             # print(input_ids.size(1),attention_msk.size(1), input_ids.device, attention_msk.device)

#             input_ids_list += input_ids
    
#     # input_ids = torch.cat(input_ids_list, dim=1)
#     # attention_mask = torch.cat(attention_mask_list, dim=1)

#     # tensor_input_ids = torch.tensor([input_ids_list])
#     # tensor_attention_mask = torch.tensor([attention_mask_list])
    
#     return {
#         'input_ids': input_ids_list,
#         'labels': labels,
#         'dataset_id': 'sft',
#         'memory_position': None,
#         'split_memory_id': None,
#         'sys_id': None
#     }

# filtered_strings = []

# # for i, string in enumerate(data2):
# #     print(i)
# def validornot(sample):

#     labels = torch.tensor(process_conversation(sample['conversations'])['labels'])
#     all_zeros = (labels == -100).all()
#     # print(all_zeros.item())
#     if all_zeros.item(): 
#         return False
        
#     else:
#         return True

# filtered_dataset = data2['train'].filter(validornot)

# # Save the filtered dataset to the specified path
# filtered_dataset.save_to_disk("/mnt/data2/jingbo/kvmemory/data/maxlen4096/sft")
# # print(f"Dataset saved to {save_path}")
      