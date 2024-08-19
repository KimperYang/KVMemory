import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

sentence = "Hello how are you"
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
logits = outputs.logits

shift_logits = logits[..., :-1, :].contiguous()
shift_labels = inputs["input_ids"][..., 1:].contiguous()

loss_fct = CrossEntropyLoss()
loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

print(f"Loss: {loss.item()}")
