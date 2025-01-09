import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

data = load_dataset("CogComp/trec")
label_dict = {0:'abbreviation', 1:'entity', 2:'description', 3:'human', 4:'location', 5:'numeric'}
# Step 1: Load the Pretrained Model and Tokenizer
model_name = "training_res/new_data/upper/checkpoint-6000"  # You can choose other models like 'gpt2-medium', 'gpt-neo-125M', etc.
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
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
    context = "<|begin_of_text|>"
    for item in data['train']:
        if all(num == num_each_class for num in num_stats) or num_demo == max_demonstration:
            break
        if num_stats[item['coarse_label']] < num_each_class:
            context += "Question: " + item['text'] + "\nType: " + label_dict[item['coarse_label']] + "\n"
            num_stats[item['coarse_label']] += 1
            num_demo += 1
    return context

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

total_num = len(data['test'])
correct_num = 0
prefix = construct_examples(data)

for idx in range(total_num):
    print(idx)

    # Step 2: Prepare the Context and Options
    context = prefix + "Question: " + data['test'][idx]['text'] + "\nType: "
    options = label_dict.values()

    # Step 3: Compute Log-Likelihoods for Each Option
    results = []
    for option in options:
        ll, length = compute_log_likelihood(context, option)
        results.append(ll / length)
        # results.append({
        #     'option': option.strip(),
        #     'log_likelihood': ll,
        #     'length': length,
        #     'avg_log_likelihood': ll / length  # For length normalization
        # })
    if  results.index(max(results)) == data['test'][idx]['coarse_label']:
        correct_num += 1
    # # Display the Results
    # for res in results:
    #     print(f"Option: {res['option']}")
    #     print(f"  Log-Likelihood: {res['log_likelihood']:.4f}")
    #     print(f"  Length: {res['length']}")
    #     print(f"  Avg Log-Likelihood (Normalized): {res['avg_log_likelihood']:.4f}\n")

print('correct_num: ', correct_num)
print('accuracy: ', correct_num / total_num)
