import os
import torch
from transformers import TrainingArguments
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import DatasetDict, load_from_disk, interleave_datasets
from accelerate import Accelerator
from src.data.dataset import custom_collate_bias
from src.training.trainer import CustomTrainerBiasAttn
from src.data.mapfunc import bias_attention_preprocessor

def main():
    batch_size_per_device = 4
    eval_num_each_set = 400
    # Prepare model and tokenizer
    
    global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    global_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16, attn_implementation='sdpa')

    new_token = ["<MEM_START>","<MEM_END>", "<MEM_SUM>"]
    global_tokenizer.add_tokens(new_token)
    global_model.resize_token_embeddings(len(global_tokenizer))

    preprocessor = bias_attention_preprocessor(
        tokenizer=global_tokenizer,
        max_len=4096
    )

    text_raw = load_from_disk("data/processed/fineweb/text")
    text_raw = text_raw.select(range(0, len(text_raw) // 2))
    text = text_raw.map(
        preprocessor.process_text,
        num_proc=64,
        remove_columns=["text", "id", "dump", "url", "date", "file_path", "language", "language_score", "token_count"],
        batched=False,
       load_from_cache_file=True
    )
    # text = text.select(range(0, 10000))
    num_text = len(text)
    text_train = text.select(range(0, int(num_text - eval_num_each_set)))
    text_eval = text.select(
        range(int(num_text - eval_num_each_set), int(num_text)),
    )

    text_mem_raw = load_from_disk("data/processed/fineweb/text_mem")
    textmem = text_mem_raw.map(
        preprocessor.process_textmem,
        num_proc=16,
        remove_columns=["text", "id", "dump", "url", "date", "file_path", "language", "language_score", "token_count"],
        batched=False,
       load_from_cache_file=True
    )
    # textmem = textmem.select(range(0, 10000))
    num_textmem = len(textmem)
    textmem_train = textmem.select(range(0, int(num_textmem - eval_num_each_set)))
    textmem_eval = textmem.select(
        range(int(num_textmem - eval_num_each_set), int(num_textmem)),
    )

    text_inst_raw = load_from_disk("data/processed/fineweb/text_inst")
    textinst = text_inst_raw.map(
        preprocessor.process_textinst,
        num_proc=16,
        remove_columns=["text", "id", "dump", "url", "date", "file_path", "language", "language_score", "token_count"],
        batched=False,
        load_from_cache_file=True
    )
    # textinst = textinst.select(range(0, 10000))
    num_textinst = len(textinst)
    textinst_train = textinst.select(range(0, int(num_textinst - eval_num_each_set)))
    textinst_eval = textinst.select(
        range(int(num_textinst - eval_num_each_set), int(num_textinst)),
    )

    sft_raw = load_from_disk("data/processed/daringanteater/sft")

    sft = sft_raw.map(
        preprocessor.process_sft,
        num_proc=64,
        remove_columns=["system", "mask", "dataset", "conversations"],
        batched=False,
#        load_from_cache_file=False
    )
    num_sft = len(sft)
    sft_train = sft.select(range(0, int(num_sft - eval_num_each_set)))
    sft_eval = sft.select(
        range(int(num_sft - eval_num_each_set), int(num_sft)),
    )

    sftmem_raw = load_from_disk("data/processed/daringanteater/sft_mem")

    sftmem = sftmem_raw.map(
        preprocessor.process_sftmem,
        num_proc=64,
        remove_columns=["system", "mask", "dataset", "conversations"],
        batched=False,
#        load_from_cache_file=False
    )
    num_sftmem = len(sftmem)
    sftmem_train = sftmem.select(range(0, int(num_sftmem - eval_num_each_set)))
    sftmem_eval = sftmem.select(
        range(int(num_sftmem - eval_num_each_set), int(num_sftmem)),
    )

    # train_dataset = textmem_train
    print("Preparing train set")
    train_dataset = interleave_datasets([sftmem_train, sft_train, textinst_train, text_train, textmem_train], probabilities=[0.25, 0.25, 0.2, 0.1, 0.2], seed=42, stopping_strategy="all_exhausted")
    print("Preparing eval set")
    eval_dataset = DatasetDict(
        {"text": text_eval,
         "textmem": textmem_eval,
         "textinst": textinst_eval,
         "sft": sft_eval,
         "sftmem": sftmem_eval}
    )
    # dataset = interleave_datasets([sftmem, sft, textinst, text, textmem, xsum], probabilities=[0.2, 0.2, 0.2, 0.1, 0.15, 0.15], seed=42, stopping_strategy="all_exhausted")

    # data_loader = DataLoader(train_dataset, batch_size= batch_size_per_device, collate_fn=custom_collate_bias, pin_memory=False)
    # eval_data_loader = DataLoader(eval_dataset, batch_size= batch_size_per_device, collate_fn=custom_collate_bias, pin_memory=False)

    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"]="kvmemory"
    os.environ["WANDB_WATCH"]="false"

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="training_res/bias_30000steps_warmup0.1_decaycosine_5e-6_full",
        report_to="wandb",
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
        remove_unused_columns=False
        # save_total_limit=3,
        # overwrite_output_dir = False
    )

    # accelerator = Accelerator()

    trainer = CustomTrainerBiasAttn(
        model=global_model,
        tokenizer=global_tokenizer,
        args=training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator = custom_collate_bias
        # data_loader = data_loader,
        # eval_data_loader = eval_data_loader
    )

    trainer.train(resume_from_checkpoint = True)

    trainer.save_model()
    global_tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()