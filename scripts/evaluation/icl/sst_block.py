import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from src.data.attention import construct_biased_attention_matrix

data = load_dataset("stanfordnlp/sst2")
label_dict = {0:'negative', 1:'positive'}

# Step 1: Load the Pretrained Model and Tokenizer
model_name = "training_res/new_data/block_prompt/checkpoint-6000"
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "/dccstor/scllm/Block-Attention/training_res/checkpoint-624"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def construct_examples(data):
    num_each_class = 10
    max_demonstration = 20
    num_demo = 0
    num_stats = [0] * len(label_dict)
    context = ["<|begin_of_text|>"]
    for item in data['train']:
        if all(num == num_each_class for num in num_stats) or num_demo == max_demonstration:
            break
        if num_stats[item['label']] < num_each_class:
            # user = f"<|start_header_id|>user<|end_header_id|>\n\nQuestion: {item['text']}\nTask: Classify this question into one of the following six types: abbreviation, entity, description, human, location, numeric.<|eot_id|>"
            # asst = f"<|start_header_id|>assistant<|end_header_id|>\n\nType: {label_dict[item['coarse_label']]}<|eot_id|>"
            user = f"<|start_header_id|>user<|end_header_id|>\n\nText: {item['sentence']}\nIs the text positive or negative?<|eot_id|>"
            asst = f"<|start_header_id|>assistant<|end_header_id|>\n\nAnswer: {label_dict[item['label']]}<|eot_id|>"
            context.append(user + asst)
            num_stats[item['label']] += 1
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

total_num = len(data['validation'])
correct_num = 0
context = construct_examples(data)

print(len(context))

biased_index = []
id_list = []
position = 0

for idx in range(len(context)):
    tem_id = tokenizer(context[idx], add_special_tokens=False).input_ids

    if "<|begin_of_text|>" not in context[idx]:

        biased_index.append([position, position + len(tem_id)])

    tem_id = torch.tensor([tem_id])
    id_list.append(tem_id)
    position = position + tem_id.size(1)

print(biased_index)
prefix_id = torch.cat(id_list, dim = 1)

for idx in range(total_num):
    print(idx)
    question = f"<|start_header_id|>user<|end_header_id|>\n\nText: {data['validation'][idx]['sentence']}\nIs the text positive or negative?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAnswer: "
    options = label_dict.values()

    results = []
    for option in options:
        question_id = tokenizer(question + option, return_tensors="pt", add_special_tokens=False).input_ids
        concat_ids = torch.cat([prefix_id, question_id], dim = 1).to(model.device)
        attention_matrices = construct_biased_attention_matrix(concat_ids.size(1), biased_index, concat_ids.size(1), model.device).unsqueeze(0).unsqueeze(0)

        ll, length = compute_log_likelihood(concat_ids, attention_matrices, option)
        results.append(ll / length)

    if  results.index(max(results)) == data['validation'][idx]['label']:
        correct_num += 1
        print('correct')

print('correct_num: ', correct_num)
print('accuracy: ', correct_num / total_num)

