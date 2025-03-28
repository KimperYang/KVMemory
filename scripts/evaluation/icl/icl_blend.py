import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
from src.utils.do_blend import append_kv, do_blend

data = load_dataset("CogComp/trec")
label_dict = {0:'abbreviation', 1:'entity', 2:'description', 3:'human', 4:'location', 5:'numeric'}

model_name = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
config = AutoConfig.from_pretrained(model_name)
model.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def construct_examples(data):
    num_each_class = 4
    max_demonstration = 20
    num_demo = 0
    num_stats = [0] * len(label_dict)
    context = ["<|begin_of_text|>"]
    for item in data['train']:
        if all(num == num_each_class for num in num_stats) or num_demo == max_demonstration:
            break
        if num_stats[item['coarse_label']] < num_each_class:
            # user = f"<|start_header_id|>user<|end_header_id|>\n\nQuestion: {item['text']}\nTask: Classify this question into one of the following six types: abbreviation, entity, description, human, location, numeric.<|eot_id|>"
            # asst = f"<|start_header_id|>assistant<|end_header_id|>\n\nType: {label_dict[item['coarse_label']]}<|eot_id|>"
            user = f"<|start_header_id|>user<|end_header_id|>\n\nCategories: abbreviation, entity, description, human, location, numeric.\nWhat category best describes: {item['text']}<|eot_id|>"
            asst = f"<|start_header_id|>assistant<|end_header_id|>\n\nAnswer: {label_dict[item['coarse_label']]}<|eot_id|>"
            context.append(user + asst)
            num_stats[item['coarse_label']] += 1
            num_demo += 1
    return context

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

total_num = len(data['test'])
correct_num = 0
context = construct_examples(data)

print(len(context))

start_pos = 0
concat_ids = []
kv_list = []
current_pos = start_pos

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

blend_kv = do_blend(model=model, old_kv=old_kv, golden_kv=golden_kv, recompute_ratio=0.18,first_layer_states=first_layer_states, position_ids=global_position_ids, config=config)

for idx in range(total_num):
    print(idx)
    # Step 2: Prepare the Context and Options
    question = f"<|start_header_id|>user<|end_header_id|>\n\nCategories: abbreviation, entity, description, human, location, numeric.\nWhat category best describes: {data['test'][idx]['text']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAnswer: "
    options = label_dict.values()

    # Step 3: Compute Log-Likelihoods for Each Option
    results = []
    for option in options:
        question_id = tokenizer(question + option, return_tensors="pt", add_special_tokens=False).input_ids
        concat_ids = torch.cat([question_id], dim = 1).to(model.device)

        ll, length = compute_log_likelihood(concat_ids, blend_kv, option)
        results.append(ll / length)
    if  results.index(max(results)) == data['test'][idx]['coarse_label']:
        correct_num += 1
        print('correct')

print('correct_num: ', correct_num)
print('accuracy: ', correct_num / total_num)
