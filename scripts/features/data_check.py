from transformers import AutoTokenizer
import json
from datasets import load_dataset
# Initialize a tokenizer (LLaMA tokenizer in this case)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
dataset = load_dataset("openwebtext")


# Your list of strings
total_samples = len(dataset['train'])

# Extract the last 90,000 samples
last_900k_samples = dataset['train'].select(range(total_samples - 900000, total_samples))

# Filter strings based on the number of tokens and print progress
def filter_strings_by_token_count(strings, min_tokens=2048):
    
    ids = tokenizer(strings['text'], add_special_tokens= False)["input_ids"]

    if(len(ids)>=min_tokens):
        return True
    
    return False

filtered_dataset = last_900k_samples.filter(filter_strings_by_token_count)

print(len(filtered_dataset))
# Save the filtered dataset to the specified path
filtered_dataset.save_to_disk("/mnt/data2/jingbo/kvmemory/data/maxlen4096/text_min_2048")
