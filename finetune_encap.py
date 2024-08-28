import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from torch.nn import CrossEntropyLoss

def generate_kv(prompt):
    tokenizer = global_tokenizer
    model = global_model
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        out = model(input_ids)
        past_key_values = out.past_key_values

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
        past_key_values = append_kv(kv_list)

        if input_ids["input_ids"].size(0) > 3500:
            return None

        # print("Inside __getitem__ or custom_collate_fn:")
        # print(f"past_key_values type: {type(past_key_values)}")
        # for i, layer in enumerate(past_key_values):
        #     print(f"Layer {i}: {type(layer)}, Length: {len(layer)}, Shape: {[k.shape for k in layer]}")

        return {
            'input_ids': input_ids["input_ids"].squeeze(0),
            'attention_mask': input_ids["attention_mask"].squeeze(0),
            'past_key_values': past_key_values
        }

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    past_key_values = [item['past_key_values'] for item in batch]

    # print("Inside __getitem__ or custom_collate_fn:")
    # print(f"past_key_values type: {type(past_key_values[0])}")
    # for i, layer in enumerate(past_key_values[0]):
    #     print(f"Layer {i}: {type(layer)}, Length: {len(layer)}, Shape: {[k.shape for k in layer]}")


    print("initial",len(past_key_values[0][0]))

    max_length = max([ids.size(0) for ids in input_ids])
    pad_token_id = global_tokenizer.pad_token_id

    padded_input_ids = torch.stack([torch.cat([ids, torch.full((max_length - len(ids),), pad_token_id)]) for ids in input_ids])
    padded_attention_mask = torch.stack([torch.cat([mask, torch.zeros(max_length - len(mask))]) for mask in attention_mask])

    return {
        'input_ids': padded_input_ids.long(),
        'attention_mask': padded_attention_mask.long(),
        'past_key_values': past_key_values
    }

def memory_seg(memory_str, seg_method='paragraph', chunk_size=25):
    memory_list = memory_str.split('.')
    clean_list = [mem for mem in memory_list if mem]
    return clean_list

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

        # Iterate over each sample in the batch
        for i in range(batch_size):
            input_id = input_ids[i].unsqueeze(0)
            attention_msk = attention_mask[i].unsqueeze(0)
            past_key_values = past_key_values_batch[i]
            num_token_masked = past_key_values[0][0].size()[2]

            input_id = input_id[:, num_token_masked:]
            outputs = global_model(input_ids=input_id, attention_mask=attention_msk, labels = input_id, past_key_values=past_key_values, use_cache=True)

            logits = outputs.logits  

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_id[..., 1:].contiguous()

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("Logits contain NaN or Inf")

            if shift_labels.min() < 0 or shift_labels.max() >= shift_logits.size(-1):
                print(f"Invalid label values: {shift_labels}")

            loss_fct = CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            losses = losses.view(shift_logits.size(0), -1)

            mask = attention_mask[i, 1:].clone()
            mask = mask[num_token_masked:]

            masked_losses = losses * mask
            final_loss = masked_losses.sum() / mask.sum()
            # print("loss:", final_loss)
            batch_loss += final_loss

        batch_loss /= batch_size
        return batch_loss
        # input_ids = inputs["input_ids"]
        # attention_mask = inputs["attention_mask"]
        # past_key_values = inputs["past_key_values"]

        # print("Inside compute_loss:")
        # print(input_ids)
        # print(f"past_key_values type: {type(past_key_values)}")
        # print(len(past_key_values))
        # for i, layer in enumerate(past_key_values[0]):
        #     print(f"Layer {i}: {type(layer)}, Length: {len(layer)}, Shape: {[k.shape for k in layer]}")

        # outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, past_key_values=past_key_values, use_cache=True)
        # logits = outputs.logits

        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_labels = input_ids[..., 1:].contiguous()

        # loss_fct = CrossEntropyLoss(reduction='none')
        # losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # losses = losses.view(shift_logits.size(0), -1)

        # mask = attention_mask[:, 1:].clone()
        # masked_losses = losses * mask
        # final_loss = masked_losses.sum() / mask.sum()

        # return (final_loss, outputs) if return_outputs else final_loss

data = [
    "A train whistle blew in the distance. The station was bustling with activity. People hurried to catch their trains. The conductor signaled the departure. The journey was about to begin.",
    "The artist mixed the paint carefully. Each stroke was deliberate and precise. The canvas slowly came to life. Colors danced under the light. The masterpiece was nearly complete.",
    "The mountain stood tall and majestic. Clouds hovered around its peak. A sense of awe filled the air. The hiker paused to take in the view. It was a moment of serenity.",
    "The clock ticked loudly in the empty room. A sense of anticipation hung in the air. She paced back and forth. The phone finally rang. Her heart skipped a beat.",
    "The cat sat on the windowsill. It watched the birds outside. The sun was shining brightly. A gentle breeze blew through the room. The day felt peaceful and calm.",
    "John took a sip of his coffee. He opened the newspaper. The headlines caught his attention. Outside, the city was waking up. He sighed, preparing for another busy day.",
    "The child ran through the park. Laughter echoed in the air. A kite flew high above. The grass was soft underfoot. It was a perfect summer afternoon.",
    "She picked up the old book. Dust rose as she turned the pages. Memories flooded back. The past seemed closer than ever. She smiled at the thought.",
    "The rain started to fall. Drops tapped against the window. The streetlights reflected on the wet pavement. He pulled his coat tighter. The night was quiet and cold.",
    "The chef prepared the ingredients. The kitchen smelled of fresh herbs. The pan sizzled on the stove. Colors and flavors blended perfectly. Dinner was almost ready."
]

dataset = CustomDataset(global_tokenizer, data)
data_loader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn,pin_memory=False)

training_args = TrainingArguments(
    output_dir="/mnt/data/jingbo/kv_dump",
    per_device_train_batch_size=2,
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
