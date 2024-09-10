import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from src.utils.cache import generate_kv_with_id, append_kv

class CustomDataset(Dataset):
    def __init__(self, tokenizer, data, model):
        self.tokenizer = tokenizer
        self.data = data
        self.model = model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #tokenize the raw data
        tokenized = self.tokenizer(self.data[idx], return_tensors="pt")  
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        #filter the samples which are too short
        num_tokens = input_ids.shape[1]

        if num_tokens < 1000:
            print("This sample is too short")
            return None

        #Here defines the max length
        # input_ids = input_ids[:, :1000]
        # attention_mask = attention_mask[:, :1000]
        input_ids = input_ids[:, :2048] 
        attention_mask = attention_mask[:, :2048]

        #Use the first 504/1000 tokens to calculate KV, the first token is <s>
        memory_ids = input_ids[:, 1:505]
        # memory_ids = input_ids[:, 1:1001]

        #The list of memory KV cache is initiated with the KV cache of <s>
        kv_list = [generate_kv_with_id(self.model, self.tokenizer("", return_tensors="pt").input_ids)]

        #Split the 504 memory ids for ten pieces of memory
        split_input_ids = torch.split(memory_ids, 51, dim=1)
        # split_input_ids = torch.split(memory_ids, 100, dim=1)
        
        for k in range(len(split_input_ids)):

            kv_cache = generate_kv_with_id(self.model, split_input_ids[k])
            kv_list.append(kv_cache)

        #Concatenate KV
        past_key_values =  append_kv(kv_list)

        #Release the memory of KV list
        del kv_list

        #Use 505 ids for computing loss
        # remaining_ids = input_ids[:, 1001:]
        remaining_ids = input_ids[:, 505:]
        # print(remaining_ids.size(1))
        return {
            'input_ids': remaining_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values
        }

def custom_collate_fn(batch):
    #Filter the None item
    batch = [item for item in batch if item is not None]
    
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    past_key_values = [item['past_key_values'] for item in batch]

    max_length = max([ids.size(1) for ids in input_ids])

    if max_length > 4000:
        max_length = 4000

    padded_input_ids = torch.stack([torch.cat([ids, torch.zeros(max_length - ids.size(1), dtype=torch.int64).unsqueeze(0)], dim = 1) for ids in input_ids])
    padded_attention_mask = torch.stack([torch.cat([mask, torch.zeros(505 + max_length - mask.size(1), dtype=torch.int64).unsqueeze(0)], dim = 1) for mask in attention_mask])

    # No padding is needed here, for the input_ids are at same lengths
    # padded_input_ids = torch.stack(input_ids)
    # padded_attention_mask = torch.stack(attention_mask)

    # torch.cuda.empty_cache()

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'past_key_values': past_key_values
    }

# class CustomDataCollator:
#     def __call__(self, features):
#         return custom_collate_fn(features)

def load_data(index):

    #Load the last number of index data in openwebtext as training data
    dataset = load_dataset("openwebtext")

    # print(type(dataset['train'][-index:]['text']))

    return dataset['train'][-index:]['text']
