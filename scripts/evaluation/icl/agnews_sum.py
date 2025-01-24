import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from src.data.attention import construct_biased_attention_matrix

data = load_dataset("sh0416/ag_news")
label_dict = {1:'World', 2:'Sports', 3:'Business', 4:'Sci/Tech'}

# Step 1: Load the Pretrained Model and Tokenizer
# model_name = "/mnt/data/jingbo/kv_dump_combine_mix5_30000steps_warmup0.1_decaycosine_5e-6_full/checkpoint-30000"
model_name = "training_res/sum/sum_5_new_mix_bsz64/checkpoint-6000"
reencode_num = 5
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

mem_start = 128054
mem_end = 128055
special_start_token = 128011

def construct_examples(data):
    num_each_class = 5
    max_demonstration = 20
    num_demo = 0
    num_stats = [0] * len(label_dict)
    context = ["<|begin_of_text|>"]
    for item in data['train']:
        if all(num == num_each_class for num in num_stats) or num_demo == max_demonstration:
            break
        if num_stats[item['label'] - 1] < num_each_class:
            # user = f"<|start_header_id|>user<|end_header_id|>\n\nQuestion: {item['text']}\nTask: Classify this question into one of the following six types: abbreviation, entity, description, human, location, numeric.<|eot_id|>"
            # asst = f"<|start_header_id|>assistant<|end_header_id|>\n\nType: {label_dict[item['coarse_label']]}<|eot_id|>"
            user = f"<|start_header_id|>user<|end_header_id|>\n\nCategories: World, Sports, Business, Sci/Tech.\nWhat category best describes:\nTitle:{item['title']}\nDescription:{item['description']}<|eot_id|>"
            asst = f"<|start_header_id|>assistant<|end_header_id|>\n\nAnswer: {label_dict[item['label']]}<|eot_id|>"
            context.append(user + asst)
            num_stats[item['label'] - 1] += 1
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

# total_num = len(data['test'])
total_num = 250
correct_num = 0
context = construct_examples(data)

print(context)

biased_index = []
id_list = []
position = 0

for idx in range(len(context[:-2])):
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
prefix_id = torch.cat(id_list, dim = 1)

for idx in range(total_num):
    print(idx)
    # Step 2: Prepare the Context and Options
    question = "".join(context[-2:]) + f"<|start_header_id|>user<|end_header_id|>\n\nCategories: World, Sports, Business, Sci/Tech.\nWhat category best describes:\nTitle:{data['test'][idx]['title']}\nDescription:{data['test'][idx]['description']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAnswer: "
    options = label_dict.values()
    # print(question)
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
    if  results.index(max(results)) + 1 == data['test'][idx]['label']:
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