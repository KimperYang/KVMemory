import torch

# NEG_INF = -1e15
NEG_INF = -float("inf")


def construct_biased_attention_matrix(
    seq_len,
    biased_ranges,
    max_len,
    device,
    left_padding=False,
):
    """
    Constructs a padded biased attention matrix.

    Parameters:
    - seq_len: The actual sequence length of the input.
    - biased_ranges: List of [start, end] indices defining biased position ranges.
    - max_len: The maximum sequence length for padding.

    Returns:
    - A numpy array representing the padded biased attention matrix.
    """
    # Initialize the attention matrix with -inf for masking
    attention_matrix = torch.triu(torch.full((max_len, max_len), float('-inf'), dtype=torch.bfloat16, device = device), diagonal= 1)

    if biased_ranges is not None:
        for range in biased_ranges:
            i = range[0]
            j = range[1]

            attention_matrix[i : j, 0 : i] = float('-inf')

    if left_padding:
        attention_matrix[:max_len-seq_len, :] = float('-inf')
        attention_matrix[: , :max_len-seq_len] = float('-inf')
    else:
        attention_matrix[seq_len :, :] = float('-inf')
        attention_matrix[: ,seq_len :] = float('-inf')

    if  attention_matrix.max() != 0:
        print("wrong", seq_len, biased_ranges, max_len)
        print(attention_matrix)

    return attention_matrix

def construct_biased_attention_matrix_v2(
    seq_len,
    biased_ranges,
    max_len,
    device,
    left_padding=False,
):
    """
    Fill in the masked positions with `NEG_INF` instead of `float("-inf)`
    """
    dtype = torch.bfloat16
    # min_type = torch.finfo(dtype).min
    # Initialize the attention matrix with -inf for masking
    attention_matrix = torch.triu(torch.full((max_len, max_len), NEG_INF, dtype=torch.bfloat16, device = device), diagonal= 1)

    if biased_ranges is not None:
        for range in biased_ranges:
            i = range[0]
            j = range[1]

            attention_matrix[i : j, 0 : i] = NEG_INF

    if left_padding:
        attention_matrix[:max_len-seq_len, :] = NEG_INF
        attention_matrix[: , :max_len-seq_len] = NEG_INF
    else:
        attention_matrix[seq_len :, :] = NEG_INF
        attention_matrix[: ,seq_len :] = NEG_INF

    return attention_matrix

