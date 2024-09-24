from transformers import AutoTokenizer
import torch

# Initialize the tokenizer (replace with your model's tokenizer if different)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# print(tokenizer("", return_tensors="pt").input_ids)
# Example tensor input_ids
ids = torch.tensor([[    1,   518, 25580, 29962,  3532, 14816, 29903,  6778,    13,  3492,
         29915,   276,   385, 20255,  1058,  1234,   278,  1139,   411,   278,
          7134,  4944,   297,   278,  9508,    13, 29966,   829, 14816, 29903,
          6778,    13,    13, 15043,   518, 29914, 25580, 29962, 29871,  6324,
         29892,   920,   526,   366, 29973,   259,     2,     1,   518, 25580,
         29962,  7197,   518, 29914, 25580, 29962, 29871,  7197]])

ids2 = torch.tensor([[    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,  6324,
         29892,   920,   526,   366, 29973,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,  7197]])

# print(ids.size(1))
# # Convert tensor to list of IDs
id_list2 = ids2.squeeze().tolist()
id_list = ids.squeeze().tolist()  # Remove any extra dimensions and convert to a list

# # Convert IDs to tokens
tokens = tokenizer.convert_ids_to_tokens(id_list)
tokens2 = tokenizer.convert_ids_to_tokens(id_list2)
print("Token list:", tokens)
print("Token list:", tokens2)
