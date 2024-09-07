import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from torch.nn import CrossEntropyLoss

def print_gpu_usage():
    # Check if CUDA is available
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated()
        gpu_memory_reserved = torch.cuda.memory_reserved()
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
        
        print(f"GPU memory allocated: {gpu_memory_allocated / (1024 ** 3):.2f} GB")
        print(f"GPU memory reserved: {gpu_memory_reserved / (1024 ** 3):.2f} GB")
        print(f"Total GPU memory: {gpu_memory_total / (1024 ** 3):.2f} GB")
    else:
        print("CUDA is not available.")

def generate_kv_with_id(input_ids):
    model = global_model
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        out = model(input_ids)
        past_key_values = out.past_key_values

    return past_key_values

def append_kv(kv_list):
    if not kv_list:
        raise ValueError("kv_list is empty. It must contain at least one past_key_values list.")
    
    num_layers = len(kv_list[0])
    concatenated_past_key_values = ()

    for layer in range(num_layers):
        keys_list = [kv[layer][0].detach() for kv in kv_list]
        values_list = [kv[layer][1].detach() for kv in kv_list]

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
        tokenized = self.tokenizer(self.data[idx], return_tensors="pt")
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        num_tokens = input_ids.shape[1]

        if num_tokens < 1000:
            return None

        input_ids = input_ids[:, :1000]
        attention_mask = attention_mask[:, :1000]

        memory_ids = input_ids[:, 1:505]

        kv_list = [generate_kv_with_id(global_tokenizer("", return_tensors="pt").input_ids)]

        split_input_ids = torch.split(memory_ids, 51, dim=1)
        
        for k in range(len(split_input_ids)):
            # print(k)
            kv_cache = generate_kv_with_id(split_input_ids[k])
            kv_list.append(kv_cache)

        past_key_values =  append_kv(kv_list)
        print("appended")
        print_gpu_usage()

        del kv_list

        remaining_ids = input_ids[:, 505:]

        print_gpu_usage()

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

    print(input_ids[0].size(1))
    max_length = max([ids.size(1) for ids in input_ids])
    pad_token_id = global_tokenizer.pad_token_id

    padded_input_ids = torch.stack([torch.cat([ids, torch.full((max_length - len(ids),), pad_token_id).unsqueeze(0)], dim = 1) for ids in input_ids])
    padded_attention_mask = torch.stack([torch.cat([mask, torch.zeros(max_length - len(mask)).unsqueeze(0)], dim = 1) for mask in attention_mask])

    torch.cuda.empty_cache()

    return {
        'input_ids': padded_input_ids.long(),
        'attention_mask': padded_attention_mask.long(),
        'past_key_values': past_key_values
    }

global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
global_tokenizer.pad_token = global_tokenizer.eos_token

global_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16)
global_model.to("cuda")

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)

global_model = get_peft_model(global_model, config)

class CustomDataCollator:
    def __call__(self, features):
        return custom_collate_fn(features)

class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        return data_loader

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        past_key_values_batch = inputs["past_key_values"]

        # if input_ids is None: 
        #     return 0

        batch_loss = 0
        batch_size = input_ids.size(0)
        print("batch size:", batch_size)
        # Iterate over each sample in the batch
        for i in range(batch_size):
            input_id = input_ids[i]
            attention_msk = attention_mask[i]
            past_key_values = past_key_values_batch[i]

            outputs = global_model(input_ids=input_id, attention_mask=attention_msk, labels = input_id, past_key_values=past_key_values, use_cache=True)

            logits = outputs.logits  

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_id[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            losses = losses.view(shift_logits.size(0), -1)

            mask = attention_msk[0, 1:].clone()
            mask = mask[505:]

            masked_losses = losses * mask
            final_loss = masked_losses.sum() / mask.sum()
            # print("loss:", final_loss)
            batch_loss += final_loss

        batch_loss /= batch_size

        torch.cuda.empty_cache()
        return batch_loss

def load_data(index):

    dataset = load_dataset("openwebtext")

    return dataset['train'][-index:]['text']

num_data_used = 10000
data = load_data(num_data_used)

dataset = CustomDataset(global_tokenizer, data)
data_loader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn,pin_memory=False)

training_args = TrainingArguments(
    output_dir="/mnt/data/jingbo/kv_dump",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_dir="/mnt/data/jingbo/logs",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    gradient_accumulation_steps=4
)

trainer = CustomTrainer(
    model=global_model,
    args=training_args
)

trainer.train()

global_model.save_pretrained("/mnt/data/jingbo/kv_dump")
global_tokenizer.save_pretrained("/mnt/data/jingbo/kv_dump")
