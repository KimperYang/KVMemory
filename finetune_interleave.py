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
from src.data.dataset import CustomDatasetCombine, custom_collate_combine
from src.training.trainer import CustomTrainerSpecial
from src.data.mapfunc import multi_kv_preprocessor

def main():
    # Prepare model and tokenizer
    global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    global_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.bfloat16, use_flash_attention_2=True)

    new_token = "<MEM>"
    global_tokenizer.add_tokens([new_token])
    global_model.resize_token_embeddings(len(global_tokenizer))

    config = LoraConfig(
        r= 64,
        lora_alpha=32,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        modules_to_save=["embed_tokens","lm_head"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM
    )

    global_model = get_peft_model(global_model, config)

    data1 = load_dataset('json', data_files='/mnt/data2/jingbo/kvmemory/filtered_strings_900000.json', split = 'train')
    preprocessor = multi_kv_preprocessor(
        tokenizer=global_tokenizer,
        max_len=2048,

    )

    data1 = data1.map(
        preprocessor.process_openwebtext,
        num_proc=32,
        remove_columns=["text"],
        batched=False,
    )

    data2 = load_from_disk("/mnt/data2/jingbo/kvmemory/filtered_strings_sft_new")
    preprocessor = multi_kv_preprocessor(
        tokenizer=global_tokenizer,
        max_len=2048,

    )

    data2 = data2.map(
        preprocessor.process_daring_anteater,
        num_proc=32,
        remove_columns=["system", "mask", "dataset", "conversations"],
        batched=False,
    )

    dataset = interleave_datasets([data1, data2], probabilities=[0.7, 0.3], seed=42, stopping_strategy="all_exhausted")
    # dataset = CustomDatasetCombine(global_tokenizer, data1, data2)
    # data_loader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_combine,pin_memory=False)

    # # set the wandb project where this run will be logged
    # os.environ["WANDB_PROJECT"]="kvmemory"
    # # os.environ["WANDB_LOG_MODEL"]="true"
    # os.environ["WANDB_WATCH"]="false"

    # # wandb.init(entity="jingboy-uc-santa-barbara",project="kvmemory", name = "kv_dump_combine_special", resume="allow")

    # # Set training arguments
    # training_args = TrainingArguments(
    #     output_dir="/mnt/data/jingbo/kv_dump_combine_special2",
    #     report_to="wandb",
    #     per_device_train_batch_size=2,
    #     # num_train_epochs=2,
    #     max_steps=30000,
    #     logging_dir="/mnt/data/jingbo/logs",
    #     logging_steps=5,
    #     save_steps=2000,
    #     gradient_accumulation_steps=4,
    #     bf16=True,
    #     learning_rate=2e-5,
    #     overwrite_output_dir = False
    # )

    # # optimizer = AdamW(global_model.parameters(), lr=1e-5)

    # # total_steps = len(data_loader) * training_args.num_train_epochs
    # # print("Total steps:",total_steps)
    # # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=10000)

    # accelerator = Accelerator()

    # trainer = accelerator.prepare(CustomTrainerSpecial(
    #     model=global_model,
    #     tokenizer=global_tokenizer,
    #     args=training_args,
    #     data_loader = data_loader,
    #     # optimizers=(optimizer, scheduler)
    # ))

    # trainer.train()

    # global_model.save_pretrained("/mnt/data/jingbo/kv_dump_combine_special2")
    # global_tokenizer.save_pretrained("/mnt/data/jingbo/kv_dump_combine_special2")

if __name__ == "__main__":
    main()