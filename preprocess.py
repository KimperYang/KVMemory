"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file configs/h100_config.yaml \
    --main_process_port 25678 block_attn_trainer.py

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/single_gpu.yaml \
    --main_process_port 25678 block_attn_trainer.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file configs/fsdp.yaml \
    --main_process_port 25678 block_attn_trainer.py
"""
from typing import Tuple

import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.data.input_preprocessor import qwen_sum_attention_preprocessor

def load_from_disk_then_process(
    data_component_name: str,
    preprocessor: qwen_sum_attention_preprocessor,
) -> Tuple[datasets.IterableDataset, datasets.Dataset]:
    """
    load the downloaded data from disk and then pair it with the preprocessor
    """
    if data_component_name in ["text", "text_mem", "text_inst"]:
        data_path = f"dataset_cache/processed/fineweb/{data_component_name}"
        if data_component_name == "text":
            preprocessor_fn = preprocessor.process_text
        else:
            raise NotImplementedError()
        remove_columns = [
            "text", "id", "dump", "url", "date",
            "file_path", "language", "language_score", "token_count",
        ]
        num_shards = 512
    elif data_component_name in ["sft", "sft_mem"]:
        data_path = f"dataset_cache/processed/daringanteater/{data_component_name}"
        if data_component_name == "sft":
            preprocessor_fn = preprocessor.process_sft
        elif data_component_name == "sft_mem":
            preprocessor_fn = preprocessor.process_sftmem
        else:
            raise NotImplementedError()
        remove_columns=["system", "mask", "dataset", "conversations"]
        num_shards = 32
    elif data_component_name in ["tulu"]:
        data_path = "dataset_cache/processed/tulu/sft"
        if data_component_name == "tulu":
            preprocessor_fn = preprocessor.process_tulu
        else:
            raise NotImplementedError()
        remove_columns=["id", "messages", "source"]
        num_shards = 32
    elif data_component_name in ["qa", "qa_mem"]:
        data_path = f"dataset_cache/processed/block_qa/{data_component_name}"
        if data_component_name == "qa":
            preprocessor_fn = preprocessor.process_qa
        elif data_component_name == "qa_mem":
            preprocessor_fn = preprocessor.process_qamem
        else:
            raise NotImplementedError()
        remove_columns=['prompt', 'question', 'answers', 'generated', 'inputs', 'documents']
        num_shards = 32
    elif data_component_name in ["xsum"]:
        data_path = f"dataset_cache/processed/xsum/{data_component_name}"
        preprocessor_fn = preprocessor.process_xsum
        remove_columns=['document', 'summary', 'id']
        num_shards = 32
    else:
        raise NotImplementedError()
    data_component: datasets.DatasetDict = datasets.load_from_disk(data_path)
    # print(data_component.cleanup_cache_files())

    if data_component_name == "text":
        streaming_train_dataset = data_component["train"][400000]
    else
        streaming_train_dataset = data_component["train"]

    training_data = streaming_train_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        num_proc=64,
        batched=False,
    )

    eval_dataset = data_component["test"]
    eval_data = eval_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        num_proc=64,
        batched=False,
    )

    return training_data, eval_data


def main():

    global_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    reencode_num = 5

    special_token_start = len(global_tokenizer)
    max_memory_num = 40
    new_special_tokens = [f"<link_{i}>" for i in range(max_memory_num * reencode_num)] + ["<mem_start>", "<mem_end>"]
    special_tokens_dict = {"additional_special_tokens": new_special_tokens}

    global_tokenizer.add_special_tokens(special_tokens_dict, replace_additional_special_tokens=False)

    mem_start = len(global_tokenizer) - 2
    mem_end = len(global_tokenizer) - 1

    assert global_tokenizer.convert_tokens_to_ids("<mem_start>") == mem_start
    assert global_tokenizer.convert_tokens_to_ids("<mem_end>") == mem_end

    print("Using special tokens: Special_token_start: ", special_token_start, " Mem_start: ", mem_start, " Mem_end: ", mem_end)

    preprocessor = qwen_sum_attention_preprocessor(
        tokenizer=global_tokenizer,
        max_len=4096,
        special_token_start=special_token_start,
        mem_start=mem_start,
        mem_end=mem_end,
        reencode_num=reencode_num,
        do_shuffle=True
    )

    text_train, text_test = load_from_disk_then_process("text", preprocessor)
    dataset = datasets.DatasetDict({'train': text_train, 'test': text_test})
    shards = {'train': 128, 'test': 4}
    dataset.save_to_disk("dataset_cache/processed/qwen_mapped/text", num_shards=shards, num_proc=128)

    tulu_train, tulu_test = load_from_disk_then_process("tulu", preprocessor)
    dataset = datasets.DatasetDict({'train': tulu_train, 'test': tulu_test})
    shards = {'train': 128, 'test': 4}
    dataset.save_to_disk("dataset_cache/processed/qwen_mapped/tulu", num_shards=shards, num_proc=128)

    qa_train, qa_test = load_from_disk_then_process("qa", preprocessor)
    dataset = datasets.DatasetDict({'train': qa_train, 'test': qa_test})
    shards = {'train': 128, 'test': 4}
    dataset.save_to_disk("dataset_cache/processed/qwen_mapped/qa", num_shards=shards, num_proc=128)

    qamem_train, qamem_test = load_from_disk_then_process("qa_mem", preprocessor)
    dataset = datasets.DatasetDict({'train': qamem_train, 'test': qamem_test})
    shards = {'train': 128, 'test': 4}
    dataset.save_to_disk("dataset_cache/processed/qwen_mapped/qa_mem", num_shards=shards, num_proc=128)

    xsum_train, xsum_test = load_from_disk_then_process("xsum", preprocessor)
    dataset = datasets.DatasetDict({'train': xsum_train, 'test': xsum_test})
    shards = {'train': 128, 'test': 4}
    dataset.save_to_disk("dataset_cache/processed/qwen_mapped/xsum", num_shards=shards, num_proc=128)

    sftmem_train, sftmem_test = load_from_disk_then_process("sft_mem", preprocessor)
    dataset = datasets.DatasetDict({'train': sftmem_train, 'test': sftmem_test})
    shards = {'train': 128, 'test': 4}
    dataset.save_to_disk("dataset_cache/processed/qwen_mapped/sft_mem", num_shards=shards, num_proc=128)

if __name__ == "__main__":
    main()
