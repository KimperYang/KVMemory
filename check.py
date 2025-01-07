from src.data.input_preprocessor import bias_attention_preprocessor, custom_collate_baseline
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

samsum = load_dataset("Samsung/samsum")

print(samsum['validation'][4].keys())

