"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file configs/h100_config.yaml \
    --main_process_port 25678 block_attn_trainer.py

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/single_gpu.yaml \
    --main_process_port 25678 block_attn_trainer.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file configs/fsdp.yaml \
    --main_process_port 25678 block_attn_trainer.py
"""
import os
from typing import Tuple

import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.data.input_preprocessor import custom_collate_bias, granite_sum_attention_preprocessor
from src.training.custom_trainer import CustomTrainerBiasAttn


def load_from_disk_then_process(
    data_component_name: str,
    preprocessor: granite_sum_attention_preprocessor,
) -> Tuple[datasets.IterableDataset, datasets.Dataset]:
    """
    load the downloaded data from disk and then pair it with the preprocessor
    """
    if data_component_name in ["text", "text_mem", "text_inst"]:
        data_path = f"dataset_cache/processed/fineweb/{data_component_name}"
        if data_component_name == "text":
            preprocessor_fn = preprocessor.process_text
        elif data_component_name == "text_mem":
            preprocessor_fn = preprocessor.process_textmem
        elif data_component_name == "text_inst":
            preprocessor_fn = preprocessor.process_textinst
        else:
            raise NotImplementedError()
        remove_columns = [
            "text", "id", "dump", "url", "date",
            "file_path", "language", "language_score", "token_count",
        ]
        num_shards = 512
        if data_component_name in ["text_mem", "text_inst"]:
            remove_columns.append("num_tokens")
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

    streaming_train_dataset = data_component["train"].to_iterable_dataset(num_shards=num_shards)
    training_data = streaming_train_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        batched=False,
    )

    eval_dataset = data_component["test"].to_iterable_dataset(num_shards=num_shards)
    eval_data = eval_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        batched=False,
    )

    return training_data, eval_data


def main():
    batch_size_per_device = 2
    reencode_num = 0

    global_tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.1-8b")
    global_model = AutoModelForCausalLM.from_pretrained(
        "ibm-granite/granite-4.1-8b",
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa'
    )

    special_token_start = len(global_tokenizer)
    max_memory_num = 40
    new_special_tokens = [f"<link_{i}>" for i in range(max_memory_num * reencode_num)] + ["<mem_start>", "<mem_end>"]
    special_tokens_dict = {"additional_special_tokens": new_special_tokens}

    global_tokenizer.add_special_tokens(special_tokens_dict, replace_additional_special_tokens=False)
    global_model.resize_token_embeddings(len(global_tokenizer))

    mem_start = len(global_tokenizer) - 2
    mem_end = len(global_tokenizer) - 1

    assert global_tokenizer.convert_tokens_to_ids("<mem_start>") == mem_start
    assert global_tokenizer.convert_tokens_to_ids("<mem_end>") == mem_end

    print("Using special tokens: Special_token_start: ", special_token_start, " Mem_start: ", mem_start, " Mem_end: ", mem_end)

    preprocessor = granite_sum_attention_preprocessor(
        tokenizer=global_tokenizer,
        max_len=4096,
        special_token_start=special_token_start,
        mem_start=mem_start,
        mem_end=mem_end,
        reencode_num=reencode_num,
        do_shuffle=True
    )

    ptr_train, ptr_eval = load_from_disk_then_process("text", preprocessor)
    sft_train, sft_eval = load_from_disk_then_process("tulu", preprocessor)
    sft_mem_train, sft_mem_eval = load_from_disk_then_process("sft_mem", preprocessor)
    qa_train, qa_eval = load_from_disk_then_process("qa", preprocessor)
    qa_mem_train, qa_mem_eval = load_from_disk_then_process("qa_mem", preprocessor)
    xsum_train, xsum_eval = load_from_disk_then_process("xsum", preprocessor)

    train_dataset = datasets.interleave_datasets(
        [sft_mem_train, sft_train, ptr_train, qa_train, qa_mem_train, xsum_train],
        probabilities=[0.25, 0.30, 0.20, 0.10, 0.10, 0.05],
        seed=42,
        stopping_strategy="all_exhausted",
    )

    eval_dataset = datasets.DatasetDict({
        "text": ptr_eval,
        "sft": sft_eval,
        "sftmem": sft_mem_eval,
        "qa": qa_eval,
        "qamem": qa_mem_eval,
        "xsum": xsum_eval
    })

    os.environ["WANDB_PROJECT"]="kvmemory"
    os.environ["WANDB_WATCH"]="false"

    training_args = TrainingArguments(
        output_dir=f"training_res/kvlink_{reencode_num}_granite_8B",
        report_to="wandb",
        run_name=f"kvlink_{reencode_num}_bsz{batch_size_per_device}_granite_8B",
        per_device_train_batch_size= batch_size_per_device,
        max_steps=6000,
        logging_dir="training_res/logs",
        logging_steps=10,
        save_steps=3000,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        bf16=True,
        learning_rate=5e-6,
        do_eval=False,
        gradient_checkpointing=True,
        save_total_limit=1,
        # overwrite_output_dir = False
        remove_unused_columns=False,
        # split_batches=True,
        dispatch_batches=False,
        # eval_on_start=True,
        seed = 42
    )

    trainer = CustomTrainerBiasAttn(
        model=global_model,
        tokenizer=global_tokenizer,
        args=training_args,
        train_dataset = train_dataset,
        # eval_dataset = eval_dataset,
        data_collator = custom_collate_bias
    )

    trainer.train()

if __name__ == "__main__":
    main()
