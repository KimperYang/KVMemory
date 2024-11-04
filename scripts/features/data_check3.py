from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
import re
data = load_dataset('EdinburghNLP/xsum')['train']
# data = load_from_disk("/mnt/data2/jingbo/kvmemory/data/maxlen4096/xsum_min5sentences")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
print(len(data))
max_length = 4096

def validornot(sample):
    token_len = len(tokenizer(sample['document'])['input_ids'] + tokenizer(sample['summary'])['input_ids'])
    if token_len < 500 or token_len > 3500:
        return False
    
    paragraphs = sample['document'].split('\n')
    if len(paragraphs) < 5:
        return False
    # sentences = re.split(r'[.!?]', sample['document'])
    
    # # Removing empty strings from the list
    # sentences = [sentence for sentence in sentences if sentence.strip()]
    # if len(sentences) < 5:
    #     return False
    return True

filtered_dataset = data.filter(validornot)
# Save the filtered dataset to the specified path
filtered_dataset.save_to_disk("/mnt/data2/jingbo/kvmemory/data/maxlen4096/xsum_min5paragraphs")
print(len(filtered_dataset))
