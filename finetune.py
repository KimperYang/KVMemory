import torch
from transformers import TrainingArguments
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from src.data.dataset import CustomDataset, load_data, custom_collate_fn
from src.training.trainer import CustomTrainer

# Prepare model and tokenizer
global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
global_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16)
global_model.to("cuda")

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)

global_model = get_peft_model(global_model, config)

# Load data
num_data_used = 20000
data = load_data(num_data_used)
dataset = CustomDataset(global_tokenizer, data, global_model)
data_loader = DataLoader(dataset, batch_size=8, collate_fn=custom_collate_fn,pin_memory=False)

# Set training arguments
training_args = TrainingArguments(
    output_dir="/mnt/data/jingbo/kv_dump",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_dir="/mnt/data/jingbo/logs",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    gradient_accumulation_steps=4
)

trainer = CustomTrainer(
    model=global_model,
    args=training_args,
    data_loader = data_loader
)

trainer.train()

global_model.save_pretrained("/mnt/data/jingbo/kv_dump")
global_tokenizer.save_pretrained("/mnt/data/jingbo/kv_dump")

trainer.save_training_curve("/mnt/data/jingbo/kv_dump")