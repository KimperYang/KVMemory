import torch

attention_matrix = torch.triu(torch.full((3, 3), float('-inf'), dtype=torch.bfloat16, device = 'cpu'), diagonal= 1) 

print(attention_matrix[0][0])