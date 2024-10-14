import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaModel
checkpoint = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForCausalLM.from_pretrained(checkpoint)  # You may want to use bfloat16 and/or move to GPU here

tokens = tokenizer("Hello how are you <MEM>")
print(tokens)
# Add the custom token
new_token = "<MEM>"
tokenizer.add_tokens([new_token])
# tokenizer.pad_token = tokenizer.eos
tokens = tokenizer("Hello how are you <MEM>")
print(tokens)
print(tokenizer.convert_ids_to_tokens([32000]))
print(torch.tensor([[32000]]*3))

# print(tokenizer.convert_ids_to_tokens(tokens[0]))
# messages = [
#     {
#         "role": "system",
#         "content": "You are a friendly chatbot who always responds in the style of a pirate",
#     },
#     {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
#     {"role": "assistant", "content": "Six in total"},
#     {"role": "user", "content": "How many helicopters can a human eat in one sitting?"}
#  ]
# tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")

# print(tokenizer.decode(tokenized_chat[0]))

# print(tokenizer.decode(torch.tensor([518, 29914, 25580, 29962])))
# print(tokenizer.decode(torch.tensor([518, 25580, 29962])))
# print(tokenizer.decode(torch.tensor([2])))