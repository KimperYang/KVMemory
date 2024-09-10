import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from src.utils.cache import generate_kv_with_id, append_kv

class CustomDataset(Dataset):
    def __init__(self, tokenizer, data):
        self.tokenizer = tokenizer
        self.data = data

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
        input_ids = input_ids[:, :4096] 
        attention_mask = attention_mask[:, :4096]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

def custom_collate_fn(batch):
    #Filter the None item
    batch = [item for item in batch if item is not None]
    
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    # past_key_values = [item['past_key_values'] for item in batch]

    max_length = max([ids.size(1) for ids in input_ids])

    padded_input_ids = torch.cat([torch.cat([ids, torch.zeros(max_length - ids.size(1), dtype=torch.int64).unsqueeze(0)], dim = 1) for ids in input_ids], dim = 0)
    padded_attention_mask = torch.cat([torch.cat([mask, torch.zeros(max_length - mask.size(1), dtype=torch.int64).unsqueeze(0)], dim = 1) for mask in attention_mask], dim = 0)

    # No padding is needed here, for the input_ids are at same lengths
    # padded_input_ids = torch.stack(input_ids)
    # padded_attention_mask = torch.stack(attention_mask)

    # torch.cuda.empty_cache()

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask
    }

def load_data(index):

    #Load the last number of index data in openwebtext as training data
    dataset = load_dataset("openwebtext")

    return dataset['train'][-index:]['text']
