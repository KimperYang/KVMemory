"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch --config_file configs/h100x6_config.yaml \
    --main_process_port 25678 finetune_bias.py
"""

from typing import Tuple

import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.data.dataset import custom_collate_bias
from src.data.mapfunc import bias_attention_preprocessor
from src.training.trainer import CustomTrainerBiasAttn


def load_from_disk_then_process(
    data_component_name: str,
    preprocessor: bias_attention_preprocessor,
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
        remove_columns = ["text"]
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
    else:
        raise NotImplementedError()
    data_component: datasets.DatasetDict = datasets.load_from_disk(data_path)
    streaming_train_dataset = data_component["train"].to_iterable_dataset(num_shards=num_shards)
    eval_dataset = data_component["test"]

    training_data = streaming_train_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        batched=False,
    )
    eval_data = eval_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        num_proc=96,
        batched=False,
    )

    return training_data, eval_data

from typing import Tuple, Dict, Optional
import pyarrow as pa
def _infer_features_from_batch(
        batch: Dict[str, list],
        try_features: Optional[datasets.features.Features] = None
    ) -> datasets.features.Features:
    pa_table = pa.Table.from_pydict(batch)
    if try_features is not None:
        try:
            pa_table = datasets.table.table_cast(pa_table, pa.schema(try_features.type))
        except (TypeError, pa.ArrowInvalid, pa.ArrowNotImplementedError):
            pass
    return datasets.features.Features.from_arrow_schema(pa_table.schema)


def main():
    batch_size_per_device = 2

    global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    global_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16, attn_implementation='sdpa')

    new_token = ["<MEM_START>","<MEM_END>", "<MEM_SUM>"]
    global_tokenizer.add_tokens(new_token)
    global_model.resize_token_embeddings(len(global_tokenizer))

    preprocessor = bias_attention_preprocessor(
        tokenizer=global_tokenizer,
        max_len=4096
    )

    ptr_train, ptr_eval = load_from_disk_then_process("text", preprocessor)
    ptr_mem_train, ptr_mem_eval = load_from_disk_then_process("text_mem", preprocessor)
    ptr_inst_train, ptr_inst_eval = load_from_disk_then_process("text_inst", preprocessor)
    sft_train, sft_eval = load_from_disk_then_process("sft", preprocessor)
    sft_mem_train, sft_mem_eval = load_from_disk_then_process("sft_mem", preprocessor)


    features = _infer_features_from_batch(ptr_train.with_format(None)._head())
    print(features)
    print(ptr_train.info)
    print(ptr_mem_train.info)
    print(ptr_inst_train.info)
    print(sft_train.info)
    print(sft_mem_train.info)

    train_dataset = datasets.interleave_datasets(
        [sft_mem_train, sft_train, ptr_inst_train, ptr_train, ptr_mem_train],
        probabilities=[0.25, 0.25, 0.2, 0.1, 0.2],
        seed=42,
        stopping_strategy="all_exhausted",
    )
    eval_dataset = datasets.DatasetDict({
        "text": ptr_eval,
        "textmem": ptr_mem_eval,
        "textinst": ptr_inst_eval,
        "sft": sft_eval,
        "sftmem": sft_mem_eval
    })

    training_args = TrainingArguments(
        output_dir="training_res/bias_30000steps_warmup0.1_decaycosine_5e-6_full",
        # report_to="wandb",
        run_name=f"bias_30000steps_bsz{batch_size_per_device}_5e-6_full",
        per_device_train_batch_size= batch_size_per_device,
        # num_train_epochs=2,
        max_steps=30000,
        logging_dir="training_res/logs",
        logging_steps=10,
        save_steps=2000,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        bf16=True,
        learning_rate=5e-6,
        do_eval=True,
        per_device_eval_batch_size = batch_size_per_device,
        evaluation_strategy="steps",  # Add this line
        eval_steps=1000,
        save_total_limit=3,
        # overwrite_output_dir = False
    )

    trainer = CustomTrainerBiasAttn(
        model=global_model,
        tokenizer=global_tokenizer,
        args=training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator = custom_collate_bias
    )

    trainer.train()

    trainer.save_model()
    global_tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
