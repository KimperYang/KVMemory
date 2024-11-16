import torch
import datetime
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def compute_f1_token_ids(generated_sequence, ground_truth_sequence):
    # Flatten the sequences
    gen_tokens = [token for seq in generated_sequence for token in seq]
    gt_tokens = [token for seq in ground_truth_sequence for token in seq]
    
    # Count overlapping tokens (with or without considering duplicates)
    # Option 1: Considering duplicates (counts)
    from collections import Counter
    gen_token_counts = Counter(gen_tokens)
    gt_token_counts = Counter(gt_tokens)
    
    common_tokens = gen_token_counts & gt_token_counts  # Intersection: min counts
    overlap = sum(common_tokens.values())
    
    # Option 2: Ignoring duplicates (set intersection)
    # overlap = len(set(gen_tokens) & set(gt_tokens))
    
    # Compute precision and recall
    precision = overlap / len(gen_tokens) if gen_tokens else 0
    recall = overlap / len(gt_tokens) if gt_tokens else 0
    
    # Compute F1 score
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return f1

def generate_kv_with_id(model, input_ids, p_id):
    input_ids = input_ids.to(model.device)
    p_id = p_id.to(model.device)
    with torch.no_grad():
        out = model(input_ids, position_ids = p_id)
        past_key_values = out.past_key_values

    return past_key_values

def load_data(index):

    dataset = load_dataset("openwebtext")

    return dataset['train'][:index]['text']

model_name = 'meta-llama/Llama-3.2-1B-Instruct'

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

raw_data = load_data(1000)
res_list = []
for i in range(len(raw_data)):
    # Tokenize the current context
    input_ids = tokenizer.encode(raw_data[i], return_tensors='pt')
    mem_num = 10
    mem_len = 100

    if input_ids.size(1) < mem_num * mem_len + 1:
        continue
    
    input_ids = input_ids[:, :mem_num * mem_len + 1]

    past_key_values =  generate_kv_with_id(model, input_ids, torch.arange(input_ids.size(1)).unsqueeze(0))

    start_position = 550
    needle_len = 10
    needle = input_ids[:, start_position:start_position + needle_len]

    question_ids = torch.cat([tokenizer.encode("What is the continuation after", return_tensors='pt', add_special_tokens= False), needle, tokenizer.encode("?", return_tensors='pt', add_special_tokens= False)], dim =1)

    concat_ids = torch.cat([input_ids, question_ids], dim = 1)

    # Generate new tokens
    output = model.generate(
        concat_ids,
        max_new_tokens=20,        # Generate three tokens each time
        do_sample=False,          # Enable sampling to introduce variability
        temperature=None,
        top_p=1.0,
        past_key_values=past_key_values,
        use_cache=True
    )
    # Extract the newly generated tokens
    new_tokens = output[:, concat_ids.shape[-1]:]

    ground_truth = input_ids[:, start_position + needle_len : start_position + needle_len + 20]

    score = compute_f1_token_ids(new_tokens, ground_truth)
    ground_truth_text = tokenizer.decode(ground_truth[0], skip_special_tokens=False)
    generated_response = tokenizer.decode(new_tokens[0], skip_special_tokens=False)
    needle_text = tokenizer.decode(needle[0], skip_special_tokens=False)
    res = {"id": str(i),"needle": needle_text, "response": generated_response, "gold_answer": ground_truth_text, "score": score}
    res_list.append(res)
    print(res)

current_time = datetime.datetime.now()
time_str = current_time.strftime("%Y%m%d-%H%M%S")

score_sum = 0
for idx in range(len(res_list)):
    score_sum += res_list[idx]["score"]
avg_score = score_sum / len(res_list)

file_name = f"result/11-12/dict/dict_llama3.2_1B_original_{avg_score}_{time_str}.jsonl"

with open(file_name, 'w', encoding='utf-8') as f:
    for entry in res_list:
        json_line = json.dumps(entry)
        f.write(json_line + '\n')

print(f"Dumped at {file_name}")


