import torch
import torch.nn.functional as F
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from src.data.attention import construct_biased_attention_matrix

import argparse

parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
parser.add_argument('--path', type=str, required=True, help='model_path')
args = parser.parse_args()

model_name = args.path

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
def compute_log_likelihood(context, option):
    # Combine context and option
    input_text = context + option
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    # Tokenize context to find context length
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    context_length = len(context_ids)
    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids)
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
    return total_log_likelihood, option_length

# total_num = len(data['test'])
correct_num = 0
context = construct_examples(data)
prefix = "".join(context)

biased_index = []
id_list = []
position = 0

test_data = construct_testset(data)
total_num = len(test_data)

for idx in range(total_num):
    if idx % 50 == 0: print(idx) 
    question = f"<|start_header_id|>user<|end_header_id|>\n\nutterance:{test_data[idx]['text']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nintent: "
    options = label_dict.values()

    results = []
    for option in options:
        question_id = tokenizer(question + option, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

        ll, length = compute_log_likelihood(prefix+question, option)
        results.append(ll / length)

    if  results.index(max(results)) == test_data[idx]['intent']:
        correct_num += 1

print()
print('upper')
print('model: ', model_name)
print('correct_num: ', correct_num)
print('accuracy: ', correct_num / total_num)
print()
