import torch
import torch.nn.functional as F
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from src.data.attention import construct_biased_attention_matrix

import argparse

parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
parser.add_argument('--reencode', type=int, required=True, help='Reencode num')
args = parser.parse_args()

reencode_num = args.reencode

model_name = f"training_res/sum/sum_{reencode_num}_prompt/checkpoint-6000"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

mem_start = 128054
mem_end = 128055
special_start_token = 128011

data = load_dataset("clinc/clinc_oos", 'small')

with open("data/raw/clinic/labels.json", "r", encoding="utf-8") as f:
    label_dict = json.load(f)

print(label_dict)

intent_names = list(label_dict.values())

intent_list_str = ", ".join(intent_names)

system_prompt = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You're an assistant who classifies the intent of the utterance. "
    f"Here are all the intent classes: {intent_list_str}<|eot_id|>"
)

def construct_examples(data):
    num_each_class = 1
    max_demonstration = 20
    num_demo = 0
    num_stats = [0] * len(label_dict)
    context =[system_prompt]
    for item in data['train']:
        if all(num == num_each_class for num in num_stats) or num_demo == max_demonstration:
            break
        if num_stats[item['intent']] < num_each_class:
            user = f"<|start_header_id|>user<|end_header_id|>\n\nutterance: {item['text']}<|eot_id|>"
            asst = f"<|start_header_id|>assistant<|end_header_id|>\n\nintent: {label_dict[str(item['intent'])]}<|eot_id|>"
            context.append(user + asst)
            num_stats[item['intent']] += 1
            num_demo += 1
    return context

def construct_testset(data):
    test_data = []
    num_each_class = 2
    max_demonstration = 250
    num_demo = 0
    num_stats = [0] * len(label_dict)
    for item in data['test']:
        if all(num == num_each_class for num in num_stats) or num_demo == max_demonstration:
            break
        if num_stats[item['intent']] < num_each_class:
            test_data.append({'text':item['text'], 'intent': item['intent']})
            num_stats[item['intent']] += 1
            num_demo += 1
    return test_data
# Function to compute log-likelihood
def compute_log_likelihood(input_ids, prefix_kv, option):
    option_ids = tokenizer.encode(option, add_special_tokens=False)
    context_length = input_ids.size(1) - len(option_ids)
    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids = input_ids, past_key_values=prefix_kv, use_cache = True)
    logits = outputs.logits
    # Shift logits and labels to align
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Extract log probabilities of option tokens
    option_log_probs = log_probs[:, context_length - 1:, :]
    option_labels = shift_labels[:, context_length - 1:]
    # Gather log probabilities for actual tokens
    option_token_log_probs = option_log_probs.gather(2, option_labels.unsqueeze(-1)).squeeze(-1)
    # Sum log probabilities to get total log-likelihood
    total_log_likelihood = option_token_log_probs.sum().item()
    # Get length of the option in tokens
    option_length = option_labels.size(1)
    # print(option_length)
    return total_log_likelihood, option_length

# total_num = len(data['test'])
correct_num = 0
context = construct_examples(data)

print(context)

biased_index = []
id_list = []
position = 0

for idx in range(len(context)):
    tem_id = tokenizer(context[idx], add_special_tokens=False).input_ids

    if idx == 0:
        tem_id += [mem_start]

    mem_idx = idx - 1
    if "<|begin_of_text|>" not in context[idx]:

        for sub_idx in range(reencode_num):
            tem_id = tem_id + [special_start_token + mem_idx * reencode_num + sub_idx]

        biased_index.append([position, position + len(tem_id) - reencode_num])

    if idx == len(context):
        tem_id += [mem_end]

    tem_id = torch.tensor([tem_id])
    id_list.append(tem_id)
    position = position + tem_id.size(1)

print(biased_index)
prefix_id = torch.cat(id_list, dim = 1).to(model.device)
test_data = construct_testset(data)
print(test_data)

total_num = len(test_data)
attention_matrix = construct_biased_attention_matrix(prefix_id.size(1), biased_index, prefix_id.size(1), model.device).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    outputs = model(input_ids = prefix_id, attention_mask = attention_matrix)

prefix_kv = outputs.past_key_values

for idx in range(total_num):
    if idx % 50 == 0: print(idx) 
    question = f"<|start_header_id|>user<|end_header_id|>\n\nutterance:{test_data[idx]['text']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nintent: "
    options = label_dict.values()

    results = []
    for option in options:
        question_id = tokenizer(question + option, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

        ll, length = compute_log_likelihood(question_id, prefix_kv, option)
        results.append(ll / length)

    if  results.index(max(results)) == test_data[idx]['intent']:
        correct_num += 1

print()
print('sum')
print('model: ', model_name)
print('correct_num: ', correct_num)
print('accuracy: ', correct_num / total_num)
print()
