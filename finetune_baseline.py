import os
import torch
# import wandb
# from torch.optim import AdamW
from transformers import TrainingArguments
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, load_from_disk, interleave_datasets
from accelerate import Accelerator
from src.data.dataset import CustomDatasetCombine, custom_collate_mix
from src.training.trainer import CustomTrainerMixBaseline
from src.data.mapfunc import baseline_preprocessor

def main():
    # Prepare model and tokenizer
    model_path = "/mnt/data/jingbo/kv_dump_combine_baseline_30000steps_warmup0.1_decaycosine_5e-6_full/checkpoint-6000"
    global_tokenizer = AutoTokenizer.from_pretrained(model_path)
    global_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, use_flash_attention_2=True)

    # new_token = ["<MEM_START>","<MEM_END>", "<MEM_SUM>"]
    # global_tokenizer.add_tokens(new_token)
    # global_model.resize_token_embeddings(len(global_tokenizer))

    # config = LoraConfig(
    #     r= 64,
    #     lora_alpha=32,
    #     target_modules=["q_proj","v_proj","k_proj","o_proj"],
    #     # modules_to_save=["embed_tokens","lm_head"],
    #     lora_dropout=0.05,
    #     task_type=TaskType.CAUSAL_LM
    # )

    # global_model = get_peft_model(global_model, config)

    preprocessor = baseline_preprocessor(
        tokenizer=global_tokenizer,
        max_len=4096
    )

    text_raw = load_from_disk("/mnt/data2/jingbo/kvmemory/data/maxlen4096/text_min_2048")

    text = text_raw.map(
        preprocessor.process_text,
        num_proc=32,
        remove_columns=["text"],
        batched=False,
#        load_from_cache_file=False
    )
    
    textmem = text_raw.map(
        preprocessor.process_textmem,
        num_proc=32,
        remove_columns=["text"],
        batched=False,
#        load_from_cache_file=False
    )

    textinst = text_raw.map(
        preprocessor.process_textinst,
        num_proc=32,
        remove_columns=["text"],
        batched=False,
#        load_from_cache_file=False
    )
    # print(len(data1[0]['input_ids'][0]), len(data1[0]['labels'][0]), data1[0]['labels'][0].count(-100), len(data1[0]['memory_position_batch']), len(data1[0]['memory_position_batch'][0]))
    # print(data1[0]['labels'][0].count(-100)-(len(data1[0]['memory_position_batch']) * len(data1[0]['memory_position_batch'][0])))
    # print(data1[1]['labels'][0].count(-100)-(len(data1[1]['memory_position_batch']) * len(data1[1]['memory_position_batch'][0])))
    sft_raw = load_from_disk("/mnt/data2/jingbo/kvmemory/data/maxlen4096/sft")

    sft = sft_raw.map(
        preprocessor.process_sft,
        num_proc=32,
        remove_columns=["system", "mask", "dataset", "conversations"],
        batched=False,
#        load_from_cache_file=False
    )

    sftmem_raw = load_from_disk("/mnt/data2/jingbo/kvmemory/data/maxlen4096/sftmem_new")

    sftmem = sftmem_raw.map(
        preprocessor.process_sftmem,
        num_proc=32,
        remove_columns=["system", "mask", "dataset", "conversations"],
        batched=False,
#        load_from_cache_file=False
    )

#     nqmem_raw = load_from_disk("/mnt/data2/jingbo/kvmemory/data/maxlen4096/nqmem")
#     nqmem = nqmem_raw.map(
#         preprocessor.process_raft_nqmem,
#         num_proc=32,
#         remove_columns=['id', 'type', 'question', 'context', 'oracle_context', 'cot_answer', 'instruction'],
#         batched=False,
# #        load_from_cache_file=False
#     )

    dataset = interleave_datasets([sftmem, sft, textinst, text, textmem], probabilities=[0.25, 0.25, 0.2, 0.1, 0.2], seed=42, stopping_strategy="all_exhausted")
    # dataset = interleave_datasets([sftmem], probabilities=[1], seed=42, stopping_strategy="all_exhausted")

    # print(dataset[0])
    # print(dataset[0]['input_ids'].size(1), dataset[0]['labels'].size(1), (dataset[0]['labels'] == -100).sum().item(), dataset[0]['memory_position_batch'].size())

    data_loader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_mix, pin_memory=False)

    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"]="kvmemory"
    # os.environ["WANDB_LOG_MODEL"]="true"
    os.environ["WANDB_WATCH"]="false"

    # wandb.init(entity="jingboy-uc-santa-barbara",project="kvmemory", name = "kv_dump_combine_special", resume="allow")

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="/mnt/data/jingbo/kv_dump_combine_baseline_30000steps_warmup0.1_decaycosine_5e-6_full",
        report_to="wandb",
        run_name="baseline_30000_warmup0.1_decaycosine_5e-6_full",
        per_device_train_batch_size=2,
        # num_train_epochs=2,
        max_steps=30000,
        logging_dir="/mnt/data/jingbo/logs",
        logging_steps=10,
        save_steps=2000,
        gradient_accumulation_steps=8,
        # gradient_checkpointing=True,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        bf16=True,
        learning_rate=5e-6,
        # save_total_limit=3,
        # overwrite_output_dir = False
    )

    # if training_args.gradient_checkpointing:
    #     global_model.gradient_checkpointing_enable()

    # optimizer = AdamW(global_model.parameters(), lr=1e-5)

    # total_steps = len(data_loader) * training_args.num_train_epochs
    # print("Total steps:",total_steps)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=10000)

    accelerator = Accelerator()

    trainer = accelerator.prepare(CustomTrainerMixBaseline(
        model=global_model,
        tokenizer=global_tokenizer,
        args=training_args,
        data_loader = data_loader,
        # optimizers=(optimizer, scheduler)
    ))

    trainer.train(resume_from_checkpoint=True)

    trainer.save_model()
    # global_model.save_pretrained("/mnt/data/jingbo/kv_dump_combine_baseline_5000steps_5e-6_full")
    global_tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()