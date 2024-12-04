import os
import torch
# import wandb
from torch.optim import AdamW
from transformers import TrainingArguments
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from accelerate import Accelerator
from src.data.dataset import CustomDataset, custom_collate_fn
from src.training.trainer import CustomTrainerConnect
from src.model.model import CustomLlamaModel

def main():
    # Prepare model and tokenizer
    global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    global_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16, use_flash_attention_2=True)
    # global_model.to("cuda")

    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM
    )

    global_model = get_peft_model(global_model, config)

    hidden_dim = global_model.config.hidden_size
    num_heads = global_model.config.num_attention_heads
    num_layers = global_model.config.num_hidden_layers
    custom_model = CustomLlamaModel(global_model, hidden_dim // num_heads, num_heads, num_layers)

    data = load_dataset('json', data_files='/mnt/data2/jingbo/kvmemory/filtered_strings_900000.json')
    data = data['train']['text']
    print("num of data:", len(data))

    dataset = CustomDataset(global_tokenizer, data)
    data_loader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn,pin_memory=False, shuffle=True)

    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"]="kvmemory"
    # os.environ["WANDB_LOG_MODEL"]="true"
    os.environ["WANDB_WATCH"]="false"

    dump_dir = "/mnt/data/jingbo/kv_dump_connect"

    # Set training arguments
    training_args = TrainingArguments(
        output_dir= dump_dir,
        report_to="wandb",
        per_device_train_batch_size=2,
        # num_train_epochs=2,
        max_steps=10000,
        logging_dir="/mnt/data/jingbo/logs",
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        gradient_accumulation_steps=4,
        bf16=True,
        learning_rate=2e-5
    )

    # optimizer = AdamW(global_model.parameters(), lr=1e-5)

    # total_steps = len(data_loader) * training_args.num_train_epochs
    # print("Total steps:",total_steps)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=10000)

    accelerator = Accelerator()
    trainer = accelerator.prepare(CustomTrainerConnect(
        model=custom_model,
        tokenizer=global_tokenizer,
        args=training_args,
        data_loader = data_loader,
        # optimizers=(optimizer, scheduler)
    ))

    trainer.train()

    custom_model.base_model.save_pretrained(dump_dir)
    global_tokenizer.save_pretrained(dump_dir)

    # Save the ParameterList for trainable key caches
    torch.save(custom_model.state_dict(), dump_dir + "/model.pt")

    # Save the ParameterList for trainable value caches
    # torch.save(custom_model.state_dict().trainable_value_caches, dump_dir + "/trainable_value_caches.pth")

if __name__ == "__main__":
    main()