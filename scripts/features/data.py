# import pandas as pd    
# jsonObj = pd.read_json(path_or_buf='nq-open-10_total_documents_gold_at_0.jsonl', lines=True)

# memory_list = []

# print(jsonObj["question"][0])
# for i in range(0,10):
#     memory_list.append(jsonObj["ctxs"][0][i]["text"])

# print(len(jsonObj))


########################
from datasets import load_dataset

# dataset = load_dataset("openwebtext")

# # Check the structure of the dataset
# print(dataset)

# sum = 0
# short = 999999
# long = 0
# # Access a specific split (e.g., 'train') and view the first few samples
# for i in range(2000):
#     length = len(dataset['train'][i]['text'].split())
#     sum = sum + length
#     if length < short:
#         short = length
#     if length > long:
#         long = length
# print(sum/2000, short, long)

data = load_dataset('json', data_files='filtered_strings_900000.json')
print(len(data['train']['text']))
###########################
# import pandas as pd    
# jsonObj = pd.read_json(path_or_buf='/home/jingbo/KVMemory/data/ifeval/input_data.jsonl', lines=True)

# memory_list = []

# print(jsonObj["prompt"][0])

# memory_list = jsonObj["prompt"][3].split(". ")

# start_token = "<s>"
# end_token = "[/INST]"
# # memory_list.insert(0, template)
# memory_list.insert(0, start_token)

# print(memory_list)

################
# from datasets import load_dataset

# dataset = load_dataset("MemGPT/MSC-Self-Instruct")

# def reorganize_dialog(data):
#     organized_dialog = []
    
#     for entry in data:
#         dialog = entry['dialog']
        
#         # Assume alternating text between PersonA and PersonB
#         for i in range(0, len(dialog) - 1, 2):
#             person_a_text = dialog[i]['text']
#             person_b_text = dialog[i + 1]['text']
            
#             # Append each exchange as a dictionary entry
#             organized_dialog.append({
#                 "User": person_a_text,
#                 "Assistant": person_b_text
#             })
    
#     return organized_dialog

# print(len(dataset["train"]["previous_dialogs"]))





# import torch
# from torch.utils.data import DataLoader, Dataset
# import torch.nn as nn
# import torch.optim as optim
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import LoraConfig, get_peft_model, TaskType
# from datasets import load_dataset
# from torch.nn import CrossEntropyLoss

# def generate_kv(prompt):

#     tokenizer = global_tokenizer
#     model = global_model

#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#     input_ids = input_ids.to(model.device)

#     with torch.no_grad():
#         out = model(input_ids)
#         past_key_values = out.past_key_values

#     #filter <s>
#     filtered_past_key_values = ()

#     for past_keys, past_values in past_key_values:

#         filtered_keys = past_keys[:, :, 1:, :] 
#         filtered_values = past_values[:, :, 1:, :] 
#         filtered_past_key_values = filtered_past_key_values + ((filtered_keys, filtered_values),)

#     input_ids = input_ids[:, 1:]

#     return input_ids, filtered_past_key_values

# def append_kv(kv_list):

#     if not kv_list:
#         raise ValueError("kv_list is empty. It must contain at least one past_key_values list.")

#     num_layers = len(kv_list[0])

#     concatenated_past_key_values = ()

#     for layer in range(num_layers):
        
#         keys_list = [kv[layer][0] for kv in kv_list]
#         values_list = [kv[layer][1] for kv in kv_list]

#         concatenated_keys = torch.cat(keys_list, dim=2)
#         concatenated_values = torch.cat(values_list, dim=2) 

#         concatenated_past_key_values = concatenated_past_key_values + ((concatenated_keys, concatenated_values),)

#     return concatenated_past_key_values

# def concat_input_id(id_list):

#     concat_input_ids = torch.cat([inp['input_ids'][:, 1:] for inp in id_list], dim=1)

#     concat_attention_mask = torch.cat([inp['attention_mask'][:, 1:] for inp in id_list], dim=1)

#     concatenated_input = {
#         'input_ids': concat_input_ids,
#         'attention_mask': concat_attention_mask
#     }

#     return concatenated_input

# class CustomDataset(Dataset):
#     def __init__(self, tokenizer, data):
#         self.tokenizer = tokenizer
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):

#         memory_list = memory_seg(self.data[idx])

#         start_token = "<s>"
#         memory_list.insert(0, start_token)

#         id_list = []
#         kv_list = []

#         for st in memory_list[:len(memory_list)//2]:
#             _, kv = generate_kv(st)
#             kv_list.append(kv)

#         for id in memory_list:
#             id = global_tokenizer(id, return_tensors="pt")
#             id_list.append(id)

#         input_ids = concat_input_id(id_list)
#         past_key_values =  append_kv(kv_list)

#         if input_ids["input_ids"].size(0) > 3500:
#             return None

#         return input_ids["input_ids"].squeeze(0), input_ids["attention_mask"].squeeze(0), past_key_values

# def custom_collate_fn(batch):
#     input_ids = [item[0] for item in batch]
#     attention_mask = [item[1] for item in batch]
#     past_key_values = [item[2] for item in batch]

#     max_length = max([ids.size(0) for ids in input_ids])

#     pad_token_id = global_tokenizer.pad_token_id

#     padded_input_ids = torch.stack([torch.cat([ids, torch.zeros(max_length - len(ids))]) for ids in input_ids])
#     padded_attention_mask = torch.stack([torch.cat([mask, torch.zeros(max_length - len(mask))]) for mask in attention_mask])

#     return padded_input_ids.long(), padded_attention_mask.long(), past_key_values

# def memory_seg(memory_str, seg_method = 'paragraph', chunk_size = 25):

#     memory_list = memory_str.split('.')

#     clean_list = []
#     for mem in memory_list:
#         if not mem == "":
#             clean_list.append(mem)
#     return clean_list

# global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# data = [
#     "A train whistle blew in the distance. The station was bustling with activity. People hurried to catch their trains. The conductor signaled the departure. The journey was about to begin.",
#     "The artist mixed the paint carefully. Each stroke was deliberate and precise. The canvas slowly came to life. Colors danced under the light. The masterpiece was nearly complete.",
#     "The mountain stood tall and majestic. Clouds hovered around its peak. A sense of awe filled the air. The hiker paused to take in the view. It was a moment of serenity.",
#     "The clock ticked loudly in the empty room. A sense of anticipation hung in the air. She paced back and forth. The phone finally rang. Her heart skipped a beat.",
#     "The cat sat on the windowsill. It watched the birds outside. The sun was shining brightly. A gentle breeze blew through the room. The day felt peaceful and calm.",
#     "John took a sip of his coffee. He opened the newspaper. The headlines caught his attention. Outside, the city was waking up. He sighed, preparing for another busy day.",
#     "The child ran through the park. Laughter echoed in the air. A kite flew high above. The grass was soft underfoot. It was a perfect summer afternoon.",
#     "She picked up the old book. Dust rose as she turned the pages. Memories flooded back. The past seemed closer than ever. She smiled at the thought.",
#     "The rain started to fall. Drops tapped against the window. The streetlights reflected on the wet pavement. He pulled his coat tighter. The night was quiet and cold.",
#     "The chef prepared the ingredients. The kitchen smelled of fresh herbs. The pan sizzled on the stove. Colors and flavors blended perfectly. Dinner was almost ready."
# ]

# dataset = CustomDataset(global_tokenizer, data)
# data_loader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn)

# global_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16, device_map="auto")

# config = LoraConfig(
#     r=8,
#     lora_alpha=32,
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=0.05,
#     task_type=TaskType.CAUSAL_LM
# )
# global_model = get_peft_model(global_model, config)

# optimizer = optim.AdamW(global_model.parameters(), lr=1e-5)
# criterion = nn.CrossEntropyLoss(reduction='none')
# accumulation_steps = 4

# num_epochs = 1
# for epoch in range(num_epochs):
#     global_model.train()
#     optimizer.zero_grad() 
#     for step, batch in enumerate(data_loader):
#         input_ids, attention_mask, past_key_values_batch = batch

#         if input_ids is None: 
#             continue

#         batch_loss = 0
#         batch_size = input_ids.size(0)

#         for i in range(batch_size):
#             input_id = input_ids[i].unsqueeze(0)
#             attention_msk = attention_mask[i].unsqueeze(0)
#             past_key_values = past_key_values_batch[i]
#             num_token_masked = past_key_values[0][0].size()[2]

#             input_id = input_id[:, num_token_masked:]
#             outputs = global_model(input_ids=input_id, attention_mask=attention_msk, labels = input_id, past_key_values=past_key_values, use_cache=True)

#             logits = outputs.logits  

#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = input_id[..., 1:].contiguous()

#             if torch.isnan(logits).any() or torch.isinf(logits).any():
#                 print("Logits contain NaN or Inf")

#             if shift_labels.min() < 0 or shift_labels.max() >= shift_logits.size(-1):
#                 print(f"Invalid label values: {shift_labels}")

#             loss_fct = CrossEntropyLoss(reduction='none')
#             losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

#             losses = losses.view(shift_logits.size(0), -1)

#             mask = attention_mask[i, 1:].clone()
#             mask = mask[num_token_masked:]

#             masked_losses = losses * mask
#             final_loss = masked_losses.sum() / mask.sum()
#             batch_loss += final_loss

#         batch_loss /= batch_size

#         batch_loss.backward()

#         torch.nn.utils.clip_grad_norm_(global_model.parameters(), max_norm=1.0)

#         if (step + 1) % accumulation_steps == 0:
#             optimizer.step()
#             optimizer.zero_grad()

#         print(f"Epoch {epoch}, Step {step + 1}, Loss: {batch_loss.item()}")

#     if (step + 1) % accumulation_steps != 0:
#         optimizer.step()
#         optimizer.zero_grad()

# global_model.save_pretrained("/mnt/data/jingbo/kv_dump")
# global_tokenizer.save_pretrained("/mnt/data/jingbo/kv_dump")
