import torch
import json
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
from src.utils.do_blend import append_kv, do_blend

data = load_dataset("clinc/clinc_oos", 'small')

with open("data/raw/clinic/labels.json", "r", encoding="utf-8") as f:
    label_dict = json.load(f)

intent_names = list(label_dict.values())

intent_list_str = ", ".join(intent_names)

system_prompt = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You're an assistant who classifies the intent of the utterance. "
    f"Here are all the intent classes: {intent_list_str}<|eot_id|>"
)

model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
config = AutoConfig.from_pretrained(model_name)
model.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

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
def compute_log_likelihood(input_ids, kv, option):
    option_ids = tokenizer.encode(option, add_special_tokens=False)
    context_length = input_ids.size(1) - len(option_ids)
    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids = input_ids, past_key_values = kv, use_cache = True)
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

total_num = 250
correct_num = 0
test_data = construct_testset(data)
context = construct_examples(data)

start_pos = 0
concat_ids = []
kv_list = []
current_pos = start_pos

print(context)

for st in context:
    input_ids = tokenizer(st, add_special_tokens=False).input_ids
    position_ids = list(range(current_pos, current_pos + len(input_ids)))
    with torch.no_grad():
        output = model(input_ids=torch.tensor([input_ids],device=model.device),
                    position_ids=torch.tensor([position_ids],device=model.device))
    kv_list.append(output.past_key_values)

    concat_ids += input_ids
    current_pos += len(input_ids)

old_kv = append_kv(kv_list)

global_position_ids = torch.tensor([list(range(start_pos, start_pos + len(concat_ids)))],device=model.device)
with torch.no_grad():
    output = model(input_ids=torch.tensor([concat_ids],device=model.device),
                    position_ids=global_position_ids,
                    output_hidden_states=True)

golden_kv = output.past_key_values
first_layer_states = output.hidden_states[2]

blend_kv = do_blend(model=model, old_kv=old_kv, golden_kv=golden_kv, recompute_ratio=0.01,first_layer_states=first_layer_states, position_ids=global_position_ids, config=config)

for idx in range(total_num):
    print(idx, correct_num)
    # Step 2: Prepare the Context and Options
    question = f"<|start_header_id|>user<|end_header_id|>\n\nutterance:{test_data[idx]['text']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nintent: "
    options = label_dict.values()

    # Step 3: Compute Log-Likelihoods for Each Option
    results = []
    for option in options:
        question_id = tokenizer(question + option, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

        ll, length = compute_log_likelihood(question_id, old_kv, option)
        results.append(ll / length)

    print(intent_names[results.index(max(results))], intent_names[test_data[idx]['intent']])
    if  results.index(max(results)) == test_data[idx]['intent']:
        correct_num += 1

print('correct_num: ', correct_num)
print('accuracy: ', correct_num / total_num)
