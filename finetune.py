import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from torch.nn import CrossEntropyLoss

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

class CustomDataset(Dataset):
    def __init__(self, tokenizer, data):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Tokenize without padding or truncation
        # encoded = self.tokenizer(
        #     self.data[idx],
        #     padding=False,
        #     truncation=False,
        #     return_tensors="pt"
        # )
        # input_ids = encoded['input_ids'].squeeze(0)
        # attention_mask = encoded['attention_mask'].squeeze(0)

        memory_list = memory_seg(self.data[idx])

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

        input_ids = concat_input_id(id_list)
        past_key_values =  append_kv(kv_list)

        if input_ids["input_ids"].size(0) > 3500:
            return None

        return input_ids["input_ids"].squeeze(0), input_ids["attention_mask"].squeeze(0), past_key_values

def custom_collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    past_key_values = [item[2] for item in batch]

    max_length = max([ids.size(0) for ids in input_ids])

    pad_token_id = global_tokenizer.pad_token_id

    # padded_input_ids = torch.stack([torch.cat([ids, torch.full((max_length - len(ids),), pad_token_id)]) for ids in input_ids])
    padded_input_ids = torch.stack([torch.cat([ids, torch.zeros(max_length - len(ids))]) for ids in input_ids])
    padded_attention_mask = torch.stack([torch.cat([mask, torch.zeros(max_length - len(mask))]) for mask in attention_mask])

    return padded_input_ids.long(), padded_attention_mask.long(), past_key_values

def memory_seg(memory_str, seg_method = 'paragraph', chunk_size = 25):

    memory_list = memory_str.split('.')

    clean_list = []
    for mem in memory_list:
        if not mem == "":
            clean_list.append(mem)
    return clean_list

def load_data(index):

    dataset = load_dataset("openwebtext")

    return dataset['train'][:index]['text']

global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
num_data_used = 2000
data = load_data(num_data_used)
dataset = CustomDataset(global_tokenizer, data)
data_loader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn)

global_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, device_map="auto")
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)
global_model = get_peft_model(global_model, config)

optimizer = optim.AdamW(global_model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss(reduction='none')
accumulation_steps = 4

num_epochs = 3 
for epoch in range(num_epochs):
    global_model.train()
    # for batch in data_loader:
    #     input_ids, attention_mask, past_key_values_batch = batch

    #     # Forward pass with KV cache, using the KV cache from each sample in the batch
    #     outputs = global_model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values_batch, use_cache=True)

    #     logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)

    #     batch_loss = 0
    #     for i in range(len(past_key_values_batch)):
    #         num_kv_tokens = past_key_values[0][0].size()[2]
    #         # logits_remaining = logits[i, num_kv_tokens:, :].unsqueeze(0)
    #         # target_remaining = input_ids[i, num_kv_tokens:].unsqueeze(0)

    #         # # Calculate the loss only for the remaining tokens
    #         # loss_mask = attention_mask[i, num_kv_tokens:].float().unsqueeze(0)
    #         # loss_values = criterion(logits_remaining.view(-1, logits_remaining.size(-1)), target_remaining.view(-1))
    #         # loss_values = loss_values.view(logits_remaining.size(0), logits_remaining.size(1))
    #         # loss = (loss_values * loss_mask).sum() / loss_mask.sum()
    #         shift_logits = logits[i, ..., :-1, :].contiguous()
    #         shift_labels = input_ids[i, 1:].contiguous()

    #         loss_fct = CrossEntropyLoss(reduction='none')
    #         losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    #         losses = losses.view(shift_logits.size(0), -1)

    #         mask = torch.ones_like(losses)
    #         mask[:, :num_token_masked] = 0 

    #         masked_losses = losses * mask
    #         final_loss = masked_losses.sum() / mask.sum()

    #         batch_loss += final_loss

    #     batch_loss /= len(past_key_values_batch)
    optimizer.zero_grad() 
    for step, batch in enumerate(data_loader):
        input_ids, attention_mask, past_key_values_batch = batch

        if input_ids is None: 
            continue

        batch_loss = 0
        batch_size = input_ids.size(0)

        # Iterate over each sample in the batch
        for i in range(batch_size):
            input_id = input_ids[i].unsqueeze(0)
            attention_msk = attention_mask[i].unsqueeze(0)
            past_key_values = past_key_values_batch[i]
            num_token_masked = past_key_values[0][0].size()[2]

            outputs = global_model(input_ids=input_id, labels = input_id, past_key_values=past_key_values, use_cache=True)

            logits = outputs.logits  

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_id[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            losses = losses.view(shift_logits.size(0), -1)

            mask = attention_mask[i, 1:].clone()
            mask[:num_token_masked] = 0
            # mask = torch.ones_like(losses)

            masked_losses = losses * mask
            final_loss = masked_losses.sum() / mask.sum()
            print("loss:", final_loss)
            batch_loss += final_loss

        batch_loss /= batch_size

        batch_loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch}, Step {step + 1}, Loss: {batch_loss.item()}")

    if (step + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

global_model.save_pretrained("/mnt/data/jingbo/kv_dump")
global_tokenizer.save_pretrained("/mnt/data/jingbo/kv_dump")
