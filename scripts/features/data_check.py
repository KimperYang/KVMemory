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
from torch.utils.data import Dataset
# data2 = load_dataset("nvidia/Daring-Anteater")
data1 = load_dataset('json', data_files='/mnt/data2/jingbo/kvmemory/filtered_strings_900000.json')
data1 = data1['train']['text']
data2 = load_from_disk("/mnt/data2/jingbo/kvmemory/filtered_strings_sft_new")
data2 = data2['conversations']

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# SFT: Chat template; Mask + Pack in preprocess; Raft, ifeval, MSC

class CustomDatasetCombine(Dataset):
    def __init__(self, tokenizer, dataset1, dataset2, max_length=2048):
        self.tokenizer = tokenizer
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.max_length = max_length
        self.len_dataset1 = len(dataset1)
        self.len_dataset2 = len(dataset2)
        self.total_length = max(self.len_dataset1, self.len_dataset2) * 2

        self.eos_token = torch.tensor([[2]])  # The eos token ID (e.g., 2)
        self.mask_token = torch.tensor([[1]])
    def __len__(self):
        return self.total_length

    def process_conversation(self, conversation):
        # Extract "Assistant" responses and mask "User" queries
        system = "[INST] <<SYS>>\nYou're an assistant who answer the question with the knowledge provided in the prompt\n<</SYS>>\n\n"
        text = system
        system_tokenized = self.tokenizer(text)
        system_input_ids = system_tokenized.input_ids
        system_attention_msk = system_tokenized.attention_mask

        mask = []
        mask.extend([0] * len(system_input_ids))
        input_ids_list = system_input_ids
        attention_mask_list = system_attention_msk

        for i in range(len(conversation)):
            
            if conversation[i]["from"] == "User":
                if i==0:
                    t = conversation[i]["value"] + " [/INST] "
                else:
                    t = "<s> [INST] " + conversation[i]["value"]  + " [/INST] " 
                    # t = " </s><s>[INST] " + conversation[i]["value"]  + " [/INST] " 
                
                tokenized = self.tokenizer(t)

                input_ids = tokenized.input_ids[1:]
                if len(mask) + len(input_ids) >= self.max_length: 
                    break
                attention_msk = tokenized.attention_mask[1:]

                mask.extend([0] * len(input_ids))
                input_ids_list += input_ids
                attention_mask_list += attention_msk

            elif conversation[i]["from"] == "Assistant":
                t = conversation[i]["value"]
                
                tokenized = self.tokenizer(t)

                input_ids = tokenized.input_ids[1:]
                attention_msk = tokenized.attention_mask[1:]
                if len(mask) + len(input_ids) > self.max_length - 1: 
                    input_ids = input_ids[:self.max_length - 1 - len(mask)]
                    attention_msk = attention_msk[:self.max_length - 1 - len(mask)]

                # # add </s>
                # input_ids = torch.cat([input_ids, self.eos_token.to(input_ids.device)], dim = 1)
                # attention_msk = torch.cat([attention_msk, self.mask_token.to(attention_msk.device)], dim = 1) 
                input_ids += [2]
                attention_msk += [1]

                # if len(mask) + input_ids.size(1) > self.max_length: 
                #     input_ids = input_ids[:, :self.max_length - len(mask)]
                #     attention_msk = attention_msk[:, :self.max_length - len(mask)]


                mask.extend([1] * len(input_ids))
                
                # print(input_ids.size(1),attention_msk.size(1), input_ids.device, attention_msk.device)

                input_ids_list += input_ids
                attention_mask_list += attention_msk
        
        # input_ids = torch.cat(input_ids_list, dim=1)
        # attention_mask = torch.cat(attention_mask_list, dim=1)

        # tensor_input_ids = torch.tensor([input_ids_list])
        # tensor_attention_mask = torch.tensor([attention_mask_list])
        
        return {
            'input_ids': input_ids_list,
            'attention_mask': attention_mask_list,
            'dataset_id': 'sft',
            'loss_mask': mask  # Mask to ignore User's tokens during loss computation
        }

    def pack_samples(self, samples):
        input_ids_list = []
        attention_mask_list = []
        total_length = 0

        for sample in samples:
            tokenized = self.tokenizer(sample, return_tensors="pt")
            input_ids = tokenized.input_ids
            attention_mask = tokenized.attention_mask
            total_length += input_ids.size(1)

            if total_length >= self.max_length:
                # Stop if total length reaches or exceeds the max_length
                break

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        # Concatenate all input_ids and attention masks
        input_ids = torch.cat(input_ids_list, dim=1)
        attention_mask = torch.cat(attention_mask_list, dim=1)

        return input_ids, attention_mask

    def __getitem__(self, idx):
        if idx % 2 == 0:
            # Take a single sample from dataset1
            sample = self.dataset1[idx // 2]
            tokenized = self.tokenizer(sample)
            input_ids = tokenized.input_ids
            attention_mask = tokenized.attention_mask
            dataset_id = 'text'

            input_ids = input_ids[:self.max_length] 
            attention_mask = attention_mask[:self.max_length]

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'dataset_id': dataset_id,
                'loss_mask': attention_mask
            }
            
        else:
            # # Pack several samples together from dataset2
            # start_idx = (idx // 2) % self.len_dataset2
            # samples_to_pack = self.dataset2[start_idx:start_idx + 5]  # Pack up to 5 samples
            # input_ids, attention_mask = self.pack_samples(samples_to_pack)
            # dataset_id = 'sft'
            conversation = self.dataset2[idx // 2]
            return self.process_conversation(conversation)
    
def custom_collate_combine(batch):
    #Filter the None item
    batch = [item for item in batch if item is not None]
    
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    dataset_ids = [item['dataset_id'] for item in batch]
    loss_masks = [item['loss_mask'] for item in batch]

    max_length = max([len(ids) for ids in input_ids])

    # print("max: ", max_length)
    # print(attention_mask[0].size(1), attention_mask[1].size(1))

    padded_input_ids = torch.cat([torch.cat([torch.tensor([ids]), torch.zeros(max_length - len(ids), dtype=torch.int64).unsqueeze(0)], dim = 1) for ids in input_ids], dim = 0)
    padded_attention_mask = torch.cat([torch.cat([torch.tensor([mask]), torch.zeros(max_length - len(mask), dtype=torch.int64).unsqueeze(0)], dim = 1) for mask in attention_mask], dim = 0)
    padded_loss_mask = torch.cat([torch.cat([torch.tensor([mask]), torch.zeros(max_length - len(mask), dtype=torch.int64).unsqueeze(0)], dim = 1) for mask in loss_masks], dim = 0)
    # print(padded_input_ids.size(), padded_attention_mask.size(), padded_loss_mask.size())
    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'dataset_id': dataset_ids,
        'loss_mask': padded_loss_mask
    }

dataset = CustomDatasetCombine(tokenizer, data1, data2)
data_loader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_combine,pin_memory=False)
for i, batch in enumerate(data_loader):
    print("Processing: ",i)
    if not len(batch['dataset_id']) == 2 :
        print(len(batch))
        break

    ids1 = batch['input_ids'][0]
    ids2 = batch['input_ids'][1]
    at1 = batch['attention_mask'][0]
    at2 = batch['attention_mask'][1]
    msk1 = batch['loss_mask'][0]
    msk2 = batch['loss_mask'][1]

    if ids1.size(0)>2048 or ids2.size(0)>2048 or at1.size(0)>2048 or at2.size(0)>2048 or msk1.size(0)>2048 or msk2.size(0)>2048 :
        print("aha")
        break

# def validornot(sample):

#     mask = process_conversation(sample['conversations'])['loss_mask']
#     all_zeros = (mask == 0).all()
#     # print(all_zeros.item())
#     if all_zeros.item(): 
#         return False
        
#     else:
#         return True

# filtered_dataset = data2['train'].filter(validornot)

# Save the filtered dataset to the specified path
# filtered_dataset.save_to_disk("/mnt/data2/jingbo/kvmemory/filtered_strings_sft_new")
# print(f"Dataset saved to {save_path}")
      
# output_file = "/mnt/data2/jingbo/kvmemory/filtered_strings_sft.json"
# with open(output_file, "w") as f:
#     json.dump(filtered_strings, f, indent=4)

# print(len(filtered_strings))