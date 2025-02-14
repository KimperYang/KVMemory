from datasets import load_dataset
from transformers import AutoTokenizer

# data=load_dataset("dgslibisey/MuSiQue", split='validation')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
# idx = 88
# prompt = ""
# for i in range(len(data[idx]['paragraphs'])):
#     prompt += f"Document [{i+1}] Title: {data[idx]['paragraphs'][i]['title']}, Text: {data[idx]['paragraphs'][i]['paragraph_text']}"

# print(len(tokenizer(prompt, add_special_tokens=False).input_ids))
# print(data[idx]['question'])
# print(data[idx]['answer'])
idx = 2284
data = load_dataset("alexfabbri/multi_news", split="validation")

summary_len = []

# for idx in range(500):

print(len(tokenizer(data[idx]['document']).input_ids), len(tokenizer(data[idx]['summary']).input_ids))
print(data[idx]['document'])
#     summary_len.append(len(tokenizer(data[idx]['summary']).input_ids))
# print(max(summary_len))
# print(sum(summary_len) / len(summary_len))