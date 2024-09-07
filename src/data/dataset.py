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
        tokenized = self.tokenizer(self.data[idx], return_tensors="pt")
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        num_tokens = input_ids.shape[1]

        if num_tokens < 1000:
            print("This sample is too long")
            return None

        input_ids = input_ids[:, :1000]
        attention_mask = attention_mask[:, :1000]

        memory_ids = input_ids[:, 1:505]

        kv_list = [generate_kv_with_id(self.model, self.tokenizer("", return_tensors="pt").input_ids)]

        split_input_ids = torch.split(memory_ids, 51, dim=1)
        
        for k in range(len(split_input_ids)):
            # print(k)
            kv_cache = generate_kv_with_id(self.model, split_input_ids[k])
            kv_list.append(kv_cache)

        past_key_values =  append_kv(kv_list)

        del kv_list

        remaining_ids = input_ids[:, 505:]

        return {
            'input_ids': remaining_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values
        }

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    past_key_values = [item['past_key_values'] for item in batch]

    # print(input_ids.size(1))
    # max_length = max([ids.size(1) for ids in input_ids])

    # padded_input_ids = torch.stack([torch.cat([ids, torch.zeros(max_length - ids.size(1)).unsqueeze(0)], dim = 1) for ids in input_ids])
    # padded_attention_mask = torch.stack([torch.cat([mask, torch.zeros(max_length - mask.size(1)).unsqueeze(0)], dim = 1) for mask in attention_mask])

    padded_input_ids = torch.stack(input_ids)
    padded_attention_mask = torch.stack(attention_mask)

    torch.cuda.empty_cache()

    return {
        'input_ids': padded_input_ids.long(),
        'attention_mask': padded_attention_mask.long(),
        'past_key_values': past_key_values
    }

# class CustomDataCollator:
#     def __call__(self, features):
#         return custom_collate_fn(features)

def load_data(index):

    dataset = load_dataset("openwebtext")

    print(type(dataset['train'][-index:]['text']))

    return dataset['train'][-index:]['text']
