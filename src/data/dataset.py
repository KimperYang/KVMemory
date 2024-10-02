import torch
from torch.utils.data import Dataset
from datasets import load_dataset
# from src.utils.cache import generate_kv_with_id, append_kv

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
        input_ids = input_ids[:, :2048] 
        attention_mask = attention_mask[:, :2048]

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
                if len(mask) + len(input_ids) > self.max_length: 
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
    print(padded_input_ids.size(), padded_attention_mask.size(), padded_loss_mask.size())
    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'dataset_id': dataset_ids,
        'loss_mask': padded_loss_mask
    }

# add position encoding in KV; edit compute loss for different situations