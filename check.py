from src.data.input_preprocessor import baseline_attention_preprocessor, custom_collate_baseline
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# qa_path = f"dataset_cache/processed/block_qa/qa"
# qa_mem_path = f"dataset_cache/processed/block_qa/qa_mem"
tl_path = f"dataset_cache/processed/tulu/sft"
tl = load_from_disk(tl_path)['train']

# qa = load_from_disk(qa_path)['train']
# qamem = load_from_disk(qa_mem_path)['train']
# remove_columns=['prompt', 'question', 'answers', 'generated', 'inputs', 'documents']
remove_columns=["id", "messages", "source"]

global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

preprocessor = baseline_attention_preprocessor(
    tokenizer=global_tokenizer,
    max_len=4096
)

tl.map(
        preprocessor.process_tulu,
        remove_columns=remove_columns,
        num_proc=96,
        batched=False
)