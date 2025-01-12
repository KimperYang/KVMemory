from src.data.input_preprocessor import sum_attention_preprocessor
from transformers import AutoTokenizer
from datasets import load_from_disk

t = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
preprocessor = sum_attention_preprocessor(tokenizer=t, max_len=4096, special_token_start=128011, mem_start=128254, mem_end=128255, reencode_num=3)

process_fn = preprocessor.process_qamem

sample = load_from_disk('dataset_cache/processed/block_qa/qa_mem')['test'][0]

print(process_fn(sample))

