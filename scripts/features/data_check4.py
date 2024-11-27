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

data2 = load_from_disk("/mnt/data2/jingbo/kvmemory/data/maxlen4096/sftmem")
# data2 = data2['train']['conversations'][:4100]
print(len(data2))
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
max_length = 4096

def process_conversation(conversation):
    
    sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an assistant who answer the question with the knowledge provided in the prompt<|eot_id|>"
    sys_tokens = tokenizer(sys, add_special_tokens= False, return_tensors= "pt")['input_ids']
    sys_len = sys_tokens.size(1)

    if len(conversation) % 2 != 0:
        if conversation[0]["from"] == "Assistant":
            conversation = conversation[1:]
        elif conversation[-1]["from"] == "User":
            conversation = conversation[:-1]
        else:
            conversation = conversation[:-1]
    
    memory_ids = []
    memory_positions = []
    current_position = sys_len
    for idx in range(0, len(conversation) - 2, 2):
        if (
            conversation[idx]["from"] == "User" and
            conversation[idx + 1]["from"] == "Assistant"
        ):
            text = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[idx]["value"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + conversation[idx + 1]["value"] + "<|eot_id|>"
            memory_tokens = tokenizer(text, add_special_tokens= False, return_tensors= "pt")['input_ids']
            memory_tokens = torch.cat([torch.tensor([[128256]]).to(memory_tokens.device), memory_tokens, torch.tensor([[128257]]).to(memory_tokens.device)], dim = 1)
            memory_ids.append(memory_tokens[0])

            mem_len = memory_tokens.size(1)
            memory_positions.append(torch.arange(current_position, current_position + mem_len))
            current_position += mem_len

    last_q = "<|start_header_id|>user<|end_header_id|>\n\n" + conversation[len(conversation) - 2]["value"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    remaining_ids = tokenizer(last_q, add_special_tokens= False, return_tensors= "pt")['input_ids']
    remaining_ids = torch.cat([torch.tensor([[128258]]), remaining_ids], dim = 1)
    labels = torch.tensor([[-100] * remaining_ids.size(1)])

    last_a = conversation[len(conversation) - 1]["value"] + "<|eot_id|>"
    answer_tokens = tokenizer(last_a, add_special_tokens= False, return_tensors= "pt")['input_ids']
    remaining_ids = torch.cat([remaining_ids, answer_tokens], dim = 1)
    labels = torch.cat([labels, answer_tokens], dim = 1)


    if(current_position + labels.size(1) >= max_length):
        return  False
    else:
        return True

filtered_strings = []

# for i, string in enumerate(data2):
#     print(i)
def validornot(sample):
    if(len(sample['conversations'])<=2 or len(sample['conversations'])%2==1):
        return False
    
    return process_conversation(sample['conversations'])

filtered_dataset = data2.filter(validornot)

# Save the filtered dataset to the specified path
filtered_dataset.save_to_disk("/mnt/data2/jingbo/kvmemory/data/maxlen4096/sftmem_new")
print(len(filtered_dataset))
# print(f"Dataset saved to {save_path}")
      
# output_file = "/mnt/data2/jingbo/kvmemory/filtered_strings_sft.json"
# with open(output_file, "w") as f:
#     json.dump(filtered_strings, f, indent=4)

# print(len(filtered_strings))