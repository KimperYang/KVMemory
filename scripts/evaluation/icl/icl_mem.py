import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

data = load_dataset("CogComp/trec")
label_dict = {0:'abbreviation', 1:'entity', 2:'description', 3:'human', 4:'location', 5:'numeric'}

# Step 1: Load the Pretrained Model and Tokenizer
# model_name = "/mnt/data/jingbo/kv_dump_combine_mix5_30000steps_warmup0.1_decaycosine_5e-6_full/checkpoint-30000"
model_name = "/mnt/data/jingbo/kv_dump_bias_50000steps_bsz64_2e-5_full/checkpoint-50000"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def generate_kv_with_id(input_ids, p_id):

    with torch.no_grad():
        out = model(input_ids, position_ids = p_id)
        past_key_values = out.past_key_values

    return past_key_values

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

def construct_examples(data):
    num_each_class = 2
    num_stats = [0] * len(label_dict)
    context = ["<|begin_of_text|>"]
    for item in data['train']:
        if all(num == num_each_class for num in num_stats):
            break
        if num_stats[item['coarse_label']] < num_each_class:
            context.append("<MEM_START>Question: " + item['text'] + "\nType: " + label_dict[item['coarse_label']] + "\n<MEM_END>")
            num_stats[item['coarse_label']] += 1
    return context

# Function to compute log-likelihood
def compute_log_likelihood(context_cache, question, option):
    # Combine context and option
    input_text = question + option
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    # Tokenize context to find context length
    context_ids = tokenizer.encode(question, add_special_tokens=False)
    context_length = len(context_ids)
    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids = input_ids, past_key_values = context_cache, use_cache = True)
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

total_num = len(data['test'])
correct_num = 0
context = construct_examples(data)

kv_list = []
idx = 0

for st in context:
    id = tokenizer(st, return_tensors="pt", add_special_tokens=False).input_ids
    position_id = torch.arange(idx, idx + id.size(1)).unsqueeze(0)
    kv = generate_kv_with_id(id.to(model.device), position_id.to(model.device))
    kv_list.append(kv)
    idx = idx + id.size(1)

context_cache = append_kv(kv_list)

for idx in range(total_num):
    print(idx)
    # Step 2: Prepare the Context and Options
    question = "<MEM_SUM>Question: " + data['test'][idx]['text'] + "\nType: "
    options = label_dict.values()

    # Step 3: Compute Log-Likelihoods for Each Option
    results = []
    for option in options:
        ll, length = compute_log_likelihood(context_cache, question, option)
        results.append(ll / length)
        # results.append({
        #     'option': option.strip(),
        #     'log_likelihood': ll,
        #     'length': length,
        #     'avg_log_likelihood': ll / length  # For length normalization
        # })
    if  results.index(max(results)) == data['test'][idx]['coarse_label']:
        correct_num += 1
        print('correct')
    # # Display the Results
    # for res in results:
    #     print(f"Option: {res['option']}")
    #     print(f"  Log-Likelihood: {res['log_likelihood']:.4f}")
    #     print(f"  Length: {res['length']}")
    #     print(f"  Avg Log-Likelihood (Normalized): {res['avg_log_likelihood']:.4f}\n")

print('correct_num: ', correct_num)
print('accuracy: ', correct_num / total_num)
