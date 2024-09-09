from transformers import AutoTokenizer
import json
from datasets import load_dataset
# Initialize a tokenizer (LLaMA tokenizer in this case)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
dataset = load_dataset("openwebtext")


# Your list of strings
string_list = dataset['train'][-900000:]['text']

# Filter strings based on the number of tokens and print progress
def filter_strings_by_token_count(strings, tokenizer, min_tokens=1000):
    filtered_strings = []
    total_strings = len(strings)
    
    for i, string in enumerate(strings):
        # Tokenize the string
        tokens = tokenizer.tokenize(string)
        
        # Print progress
        print(f"Processing {i+1}/{total_strings}: {len(tokens)} tokens found.")
        
        # Keep only strings with 1000 or more tokens
        if len(tokens) >= min_tokens:
            filtered_strings.append(string)
    
    return filtered_strings

# Filter the string list
filtered_strings = filter_strings_by_token_count(string_list, tokenizer)

# Save the filtered strings in a JSON file
output_file = "/mnt/data2/jingbo/kvmemory/filtered_strings_900000.json"
with open(output_file, "w") as f:
    json.dump(filtered_strings, f, indent=4)

print(f"Filtered strings saved to {output_file}")
