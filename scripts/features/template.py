import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaModel
checkpoint = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForCausalLM.from_pretrained(checkpoint)  # You may want to use bfloat16 and/or move to GPU here

test = "</s>"
# tokenizer.pad_token = tokenizer.eos
tokens = tokenizer(test)
print(tokens)
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

# import torch

# # Define the tensor
# tensor = torch.tensor([[1,   518, 25580, 29962,  3532, 14816, 29903,  6778,    13,  3492,
#            526,   263, 19780, 13563,  7451,  1058,  2337, 10049, 29879,   297,
#            278,  3114,   310,   263, 21625,   403,    13, 29966,   829, 14816,
#          29903,  6778,    13,    13,  5328,  1784,  1081,   293,   459,  2153,
#            508,   263,  5199, 17545,   297,   697, 16246, 29973,   518, 29914,
#          25580, 29962, 18372,   297,  3001, 29871,     2,     1,   518, 25580,
#          29962,  1128,  1784,  1081,   293,   459,  2153,   508,   263,  5199,
#          17545,   297,   697, 16246, 29973,   518, 29914, 25580, 29962]])

# # Define the start and end sequences
# start_seq = [518, 29914, 25580, 29962]
# end_seq = [2]

# # Find where the start sequence begins
# start_idx = (tensor == torch.tensor(start_seq).unsqueeze(1)).all(dim=0).nonzero(as_tuple=True)[0].item() + len(start_seq)

# # Find where the end sequence begins
# end_idx = (tensor == torch.tensor(end_seq).unsqueeze(1)).all(dim=0).nonzero(as_tuple=True)[0].item()

# result = tensor[0, start_idx:end_idx]

# print(result)
# # # <s> [INST] <<SYS>>
# # # You are a friendly chatbot who always responds in the style of a pirate
# # # <</SYS>>

# # # How many helicopters can a human eat in one sitting? [/INST]
