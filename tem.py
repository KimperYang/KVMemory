# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from torch.nn import CrossEntropyLoss
# import pandas as pd    
# import json
# import datetime
# from datasets import load_dataset
# from peft import PeftModel, PeftConfig
# from transformers import LlamaForCausalLM

# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, use_flash_attention_2=True)
# model.to('cuda')
# # sentences = ["Hello, how are you?", "I am fine, thank you!"]
# # inputs = tokenizer(sentences, return_tensors='pt', padding=True)
# # input_ids = inputs['input_ids']
# # batch_size, seq_length = input_ids.shape

# inputs = tokenizer("How are you", return_tensors='pt')
# seq_len = inputs.input_ids.size(1)
# # fullmask = torch.tensor([[[[0] * seq_len] * seq_len]])
# fullmask = torch.tensor([
#     [[[0.0, float('-inf'), float('-inf'), float('-inf')], [0.0, 0.0, float('-inf'), float('-inf')], [0.0, 0.0, 0.0, float('-inf')], [0.0, 0.0, 0.0, 0.0]]]
#     ], dtype = torch.bfloat16)
# # model(inputs.input_ids.to(model.device), fullmask.to(model.device))
# model(inputs.input_ids.to(model.device), attention_mask = torch.tensor([[0,1,1,1]]).to(model.device))
import numpy as np

def construct_biased_attention_matrix(seq_len, biased_ranges):
    
    if biased_ranges is None:
        biased_ranges = []
    # Initialize attention matrix with zeros
    attention_matrix = np.zeros((seq_len, seq_len))

    # Create a mapping from position to biased range index
    position_to_biased_range = [None] * seq_len
    for idx, (start, end) in enumerate(biased_ranges):
        for pos in range(start, end + 1):
            position_to_biased_range[pos] = idx  # Assign biased range index

    # Build the attention matrix
    for i in range(seq_len):
        biased_range_i = position_to_biased_range[i]
        for j in range(seq_len):
            if j > i:
                attention_matrix[i, j] = float('-inf')  # Causal mask
            else:
                if biased_range_i is not None:
                    # Token i is in a biased range
                    if position_to_biased_range[j] != biased_range_i:
                        attention_matrix[i, j] = float('-inf')  # Can't attend outside biased range
                    # Else, can attend to tokens within the same biased range up to position i
                else:
                    # Token i is not in any biased range
                    # Can attend to all preceding tokens (including those in biased ranges)
                    pass  # No additional masking needed

    return attention_matrix.tolist()

# Example usage:
seq_len = 6
biased_ranges = [[2, 3]]
attention_matrix1 = construct_biased_attention_matrix(seq_len, biased_ranges)
seq_len = 8
biased_ranges = [[2, 3],[6, 7]]
attention_matrix2 = construct_biased_attention_matrix(seq_len, biased_ranges)
seq_len = 6
biased_ranges = None
attention_matrix3 = construct_biased_attention_matrix(seq_len, biased_ranges)
print(attention_matrix3)

attention_matrix_list = [attention_matrix1, attention_matrix2, attention_matrix3]

def pad_attention_matrices(attention_matrices):
    """
    Pads a list of attention matrices to the maximum sequence length in the batch.

    Parameters:
    - attention_matrices: List of numpy arrays, each representing an attention matrix for a sequence.

    Returns:
    - List of padded attention matrices.
    """
    # Find the maximum sequence length in the batch
    max_seq_len = max(len(mat) for mat in attention_matrices)
    
    padded_matrices = []
    for mat in attention_matrices:
        mat = np.array(mat)
        seq_len = mat.shape[0]
        # Initialize a padded matrix filled with -inf
        padded_mat = np.full((max_seq_len, max_seq_len), float('-inf'))
        # Copy the original attention matrix into the top-left corner
        padded_mat[:seq_len, :seq_len] = mat
        padded_matrices.append([padded_mat.tolist()])
    
    return padded_matrices

print(np.array(pad_attention_matrices(attention_matrix_list)).shape)
# for mat in pad_attention_matrices(attention_matrix_list):
#     print(mat) 
#     print(type(mat))