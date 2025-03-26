from datasets import load_dataset
from transformers import AutoTokenizer

data = load_dataset("Samsung/samsum", split="train")
t = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
exceed_num = 0
for item in data:
    length = len(t(item['dialogue'], add_special_tokens=False).input_ids)
    if length > 100:
        print(length)
        exceed_num +=1

print(exceed_num)