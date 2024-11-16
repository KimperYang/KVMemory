import os
import torch
from transformers import TrainingArguments
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk, interleave_datasets
from accelerate import Accelerator
from src.data.dataset import custom_collate_bias
from src.training.trainer import CustomTrainerBiasAttn
from src.data.mapfunc import bias_attention_preprocessor

def main():
    batch_size_per_device = 2
    # Prepare model and tokenizer
    
    global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    global_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16, use_flash_attention_2=False)

    new_token = ["<MEM_START>","<MEM_END>", "<MEM_SUM>"]
    global_tokenizer.add_tokens(new_token)
    global_model.resize_token_embeddings(len(global_tokenizer))

    preprocessor = bias_attention_preprocessor(
        tokenizer=global_tokenizer,
        max_len=4096
    )

    text_raw = load_from_disk("/mnt/data2/jingbo/kvmemory/data/maxlen4096/text_min_2048")

    text = text_raw.map(
        preprocessor.process_text,
        num_proc=256,
        remove_columns=["text"],
        batched=False,
#        load_from_cache_file=False
    )
    
    textmem = text_raw.map(
        preprocessor.process_textmem,
        num_proc=256,
        remove_columns=["text"],
        batched=False,
#        load_from_cache_file=False
    )

    textinst = text_raw.map(
        preprocessor.process_textinst,
        num_proc=256,
        remove_columns=["text"],
        batched=False,
        # load_from_cache_file=False
    )
    # print(len(data1[0]['input_ids'][0]), len(data1[0]['labels'][0]), data1[0]['labels'][0].count(-100), len(data1[0]['memory_position_batch']), len(data1[0]['memory_position_batch'][0]))
    # print(data1[0]['labels'][0].count(-100)-(len(data1[0]['memory_position_batch']) * len(data1[0]['memory_position_batch'][0])))
    # print(data1[1]['labels'][0].count(-100)-(len(data1[1]['memory_position_batch']) * len(data1[1]['memory_position_batch'][0])))
    sft_raw = load_from_disk("/mnt/data2/jingbo/kvmemory/data/maxlen4096/sft")

    sft = sft_raw.map(
        preprocessor.process_sft,
        num_proc=256,
        remove_columns=["system", "mask", "dataset", "conversations"],
        batched=False,
#        load_from_cache_file=False
    )

    sftmem_raw = load_from_disk("/mnt/data2/jingbo/kvmemory/data/maxlen4096/sftmem_new")

    sftmem = sftmem_raw.map(
        preprocessor.process_sftmem,
        num_proc=256,
        remove_columns=["system", "mask", "dataset", "conversations"],
        batched=False,
#        load_from_cache_file=False
    )

#     xsum_raw = load_from_disk("/mnt/data2/jingbo/kvmemory/data/maxlen4096/xsum_min5paragraphs")
#     xsum = xsum_raw.map(
#         preprocessor.process_xsum,
#         num_proc=32,
#         remove_columns=["document", "summary", "id"],
#         batched=False,
# #        load_from_cache_file=False
#     )

#     nqmem_raw = load_from_disk("/mnt/data2/jingbo/kvmemory/data/maxlen4096/nqmem")
#     nqmem = nqmem_raw.map(
#         preprocessor.process_raft_nqmem,
#         num_proc=32,
#         remove_columns=['id', 'type', 'question', 'context', 'oracle_context', 'cot_answer', 'instruction'],
#         batched=False,
# #        load_from_cache_file=False
#     )

    dataset = interleave_datasets([sftmem, sft, textinst, text, textmem], probabilities=[0.25, 0.25, 0.2, 0.1, 0.2], seed=42, stopping_strategy="all_exhausted")
    # dataset = interleave_datasets([sftmem, sft, textinst, text, textmem, xsum], probabilities=[0.2, 0.2, 0.2, 0.1, 0.15, 0.15], seed=42, stopping_strategy="all_exhausted")

    data_loader = DataLoader(dataset, batch_size= batch_size_per_device, collate_fn=custom_collate_bias, pin_memory=False)

    # set the wandb project where this run will be logged
    # os.environ["WANDB_PROJECT"]="kvmemory"
    # os.environ["WANDB_WATCH"]="false"

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="/mnt/data/jingbo/kv_dump_bias_30000steps_warmup0.1_decaycosine_5e-6_full",
        # report_to="wandb",
        run_name="bias_30000steps_warmup0.1_decaycosine_5e-6_full",
        per_device_train_batch_size= batch_size_per_device,
        # num_train_epochs=2,
        max_steps=30000,
        logging_dir="/mnt/data/jingbo/logs",
        logging_steps=10,
        save_steps=2000,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        bf16=True,
        learning_rate=5e-6,
        # save_total_limit=3,
        # overwrite_output_dir = False
    )

    accelerator = Accelerator()

    trainer = accelerator.prepare(CustomTrainerBiasAttn(
        model=global_model,
        tokenizer=global_tokenizer,
        args=training_args,
        data_loader = data_loader
    ))

    trainer.train()

    trainer.save_model()
    global_tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()