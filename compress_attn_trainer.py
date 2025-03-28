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
from functools import partial

from src.data.input_preprocessor import custom_collate_compress, compress_attention_preprocessor
from src.training.custom_trainer import CustomTrainerCompressAttn


def load_from_disk_then_process(
    data_component_name: str,
    preprocessor: compress_attention_preprocessor,
) -> Tuple[datasets.IterableDataset, datasets.Dataset]:
    """
    load the downloaded data from disk and then pair it with the preprocessor
    """
    if data_component_name in ["text", "text_mem", "text_inst", "text_compress"]:
        data_path = f"dataset_cache/processed/fineweb/{data_component_name}"
        if data_component_name == "text":
            preprocessor_fn = preprocessor.process_text
        elif data_component_name == "text_mem":
            preprocessor_fn = preprocessor.process_textmem
        elif data_component_name == "text_inst":
            preprocessor_fn = preprocessor.process_textinst
        elif data_component_name == "text_compress":
            preprocessor_fn = preprocessor.process_pretraining_compress
            data_path = f"dataset_cache/processed/fineweb/text_mem"
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
    else:
        raise NotImplementedError()
    data_component: datasets.DatasetDict = datasets.load_from_disk(data_path)
    # print(data_component.cleanup_cache_files())

    streaming_train_dataset = data_component["train"].to_iterable_dataset(num_shards=num_shards)
    # streaming_train_dataset = data_component["train"]
    training_data = streaming_train_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        # num_proc=16,
        batched=False,
    )

    eval_dataset = data_component["test"]
    eval_data = eval_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        num_proc=16,
        batched=False,
        load_from_cache_file=False
    )

    return training_data, eval_data


def main():
    batch_size_per_device = 8
    compress_tokens = list(range(128011, 128061))

    global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    global_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
        # use_flash_attention_2=True,
    )

    preprocessor = compress_attention_preprocessor(
        tokenizer=global_tokenizer,
        max_len=4096,
        compress_tokens=compress_tokens,
        do_shuffle=True
    )

    # ptr_train, ptr_eval = load_from_disk_then_process("text", preprocessor)
    # ptr_mem_train, ptr_mem_eval = load_from_disk_then_process("text_mem", preprocessor)
    # ptr_inst_train, ptr_inst_eval = load_from_disk_then_process("text_inst", preprocessor)
    # sft_train, sft_eval = load_from_disk_then_process("tulu", preprocessor)
    # sft_mem_train, sft_mem_eval = load_from_disk_then_process("sft_mem", preprocessor)
    # qa_train, qa_eval = load_from_disk_then_process("qa", preprocessor)
    # qa_mem_train, qa_mem_eval = load_from_disk_then_process("qa_mem", preprocessor)

    # train_dataset = datasets.interleave_datasets(
    #     [sft_mem_train, sft_train, ptr_inst_train, ptr_train, ptr_mem_train, qa_train, qa_mem_train],
    #     probabilities=[0.2, 0.25, 0.1, 0.25, 0.1, 0.05, 0.05],
    #     seed=42,
    #     stopping_strategy="all_exhausted",
    # )

    # train_dataset = datasets.interleave_datasets(
    #     [sft_mem_train, sft_train, ptr_inst_train, ptr_train, ptr_mem_train],
    #     probabilities=[0.25, 0.25, 0.2, 0.1, 0.2],
    #     seed=42,
    #     stopping_strategy="all_exhausted",
    # )

    # eval_dataset = datasets.DatasetDict({
    #     "text": ptr_eval,
    #     "textmem": ptr_mem_eval,
    #     "textinst": ptr_inst_eval,
    #     "sft": sft_eval,
    #     "sftmem": sft_mem_eval,
    #     "qa": qa_eval,
    #     "qamem": qa_mem_eval
    # })

    train_dataset, eval_dataset = load_from_disk_then_process("text_compress", preprocessor)

    os.environ["WANDB_PROJECT"]="kvmemory"
    os.environ["WANDB_WATCH"]="false"

    training_args = TrainingArguments(
        output_dir=f"training_res/compress/compress_pretrain",
        report_to="wandb",
        run_name=f"compress_50_pretrain_bsz{batch_size_per_device}_5e-6",
        per_device_train_batch_size= batch_size_per_device,
        # num_train_epochs=2,
        max_steps=8000,
        logging_dir="training_res/logs",
        logging_steps=10,
        save_steps=4000,
        gradient_accumulation_steps=1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        bf16=True,
        learning_rate=5e-6,
        do_eval=True,
        per_device_eval_batch_size = batch_size_per_device,
        evaluation_strategy="steps",  # Add this line
        eval_steps=2000,
        gradient_checkpointing=True,
        # save_total_limit=3,
        # overwrite_output_dir = False
        remove_unused_columns=False,
        # split_batches=True,
        dispatch_batches=False,
        eval_on_start=True,
        seed = 42
    )

    trainer = CustomTrainerCompressAttn(
        model=global_model,
        tokenizer=global_tokenizer,
        args=training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator = partial(custom_collate_compress,compress_tokens=compress_tokens),
        num_sum_tokens=len(compress_tokens)
    )

    trainer.train()

    # trainer.save_model()
    # global_tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
