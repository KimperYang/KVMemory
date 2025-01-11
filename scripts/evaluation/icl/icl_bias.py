import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from src.data.attention import construct_biased_attention_matrix

data = load_dataset("CogComp/trec")
label_dict = {0:'abbreviation', 1:'entity', 2:'description', 3:'human', 4:'location', 5:'numeric'}

# Step 1: Load the Pretrained Model and Tokenizer
# model_name = "/mnt/data/jingbo/kv_dump_combine_mix5_30000steps_warmup0.1_decaycosine_5e-6_full/checkpoint-30000"
model_name = "training_res/new_data/bias/checkpoint-6000"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def construct_examples(data):
    num_each_class = 2
    max_demonstration = 10
    num_demo = 0
    num_stats = [0] * len(label_dict)
    context = ["<|begin_of_text|>"]
    for item in data['train']:
        if all(num == num_each_class for num in num_stats) or num_demo == max_demonstration:
            break
        if num_stats[item['coarse_label']] < num_each_class:
            user = f"<MEM_START><|start_header_id|>user<|end_header_id|>\n\nCategories: abbreviation, entity, description, human, location, numeric.\nWhat category best describes: {item['text']}<|eot_id|>"
            asst = f"<|start_header_id|>assistant<|end_header_id|>\n\nAnswer: {label_dict[item['coarse_label']]}<|eot_id|><MEM_END>"
            context.append(user + asst)
            num_stats[item['coarse_label']] += 1
            num_demo += 1
    return context

# Function to compute log-likelihood
def compute_log_likelihood(input_ids, attention_matrix, option):
    option_ids = tokenizer.encode(option, add_special_tokens=False)
    context_length = input_ids.size(1) - len(option_ids)
    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids = input_ids, attention_mask = attention_matrix, use_cache = False)
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

for ctx in context:
    print(ctx)
    print()

biased_index = []
id_list = []
idx = 0

for st in context:
    tem_id = tokenizer(st, return_tensors="pt", add_special_tokens=False).input_ids
    id_list.append(tem_id)
    if "<MEM_START>" in st:
        biased_index.append([idx, idx + tem_id.size(1)])
    idx = idx + tem_id.size(1)

prefix_id = torch.cat(id_list, dim = 1)

for idx in range(total_num):
    print(idx)
    # Step 2: Prepare the Context and Options
    question = f"<MEM_SUM><|start_header_id|>user<|end_header_id|>\n\nCategories: abbreviation, entity, description, human, location, numeric.\nWhat category best describes: {data['test'][idx]['text']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAnswer: "
    # question = "<MEM_SUM>Question: " + data['test'][idx]['text'] + "\nType: "
    options = label_dict.values()

    # Step 3: Compute Log-Likelihoods for Each Option
    results = []
    for option in options:
        question_id = tokenizer(question + option, return_tensors="pt", add_special_tokens=False).input_ids
        concat_ids = torch.cat([prefix_id, question_id], dim = 1).to(model.device)
        attention_matrices = construct_biased_attention_matrix(concat_ids.size(1), biased_index, concat_ids.size(1), model.device).unsqueeze(0).unsqueeze(0)

        ll, length = compute_log_likelihood(concat_ids, attention_matrices, option)
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
    # Display the Results
    # print(question)
    # for res in results:
    #     print(f"Option: {res['option']}")
    #     print(f"  Log-Likelihood: {res['log_likelihood']:.4f}")
    #     print(f"  Length: {res['length']}")
    #     print(f"  Avg Log-Likelihood (Normalized): {res['avg_log_likelihood']:.4f}\n")

print('correct_num: ', correct_num)
print('accuracy: ', correct_num / total_num)
