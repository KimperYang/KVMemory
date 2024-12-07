import torch


def construct_biased_attention_matrix(seq_len, biased_ranges, max_len, device):
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

    attention_matrix[seq_len :, :] = float('-inf')
    attention_matrix[: ,seq_len :] = float('-inf')

    return attention_matrix

# def construct_biased_attention_matrix(seq_len, biased_ranges):
#     if biased_ranges is None:
#         biased_ranges = []
#     # Initialize attention matrix with zeros
#     attention_matrix = np.zeros((seq_len, seq_len))

#     # Create a mapping from position to biased range index
#     position_to_biased_range = [None] * seq_len
#     for idx, (start, end) in enumerate(biased_ranges):
#         for pos in range(start, end + 1):
#             position_to_biased_range[pos] = idx  # Assign biased range index

#     # Build the attention matrix
#     for i in range(seq_len):
#         biased_range_i = position_to_biased_range[i]
#         for j in range(seq_len):
#             if j > i:
#                 attention_matrix[i, j] = float('-inf')  # Causal mask
#             else:
#                 if biased_range_i is not None:
#                     # Token i is in a biased range
#                     if position_to_biased_range[j] != biased_range_i:
#                         attention_matrix[i, j] = float('-inf')  # Can't attend outside biased range
#                     # Else, can attend to tokens within the same biased range up to position i
#                 else:
#                     # Token i is not in any biased range
#                     # Can attend to all preceding tokens (including those in biased ranges)
#                     pass  # No additional masking needed

#     return attention_matrix.tolist()

# def construct_biased_attention_matrix(seq_len, biased_ranges, max_len):
#     """
#     Constructs a padded biased attention matrix.

#     Parameters:
#     - seq_len: The actual sequence length of the input.
#     - biased_ranges: List of [start, end] indices defining biased position ranges.
#     - max_len: The maximum sequence length for padding.

#     Returns:
#     - A numpy array representing the padded biased attention matrix.
#     """
#     if biased_ranges is None:
#         biased_ranges = []
#     # Initialize the attention matrix with -inf for masking
#     attention_matrix = np.full((max_len, max_len), float('-inf'))

#     # Create a mapping from position to biased range index
#     position_to_biased_range = [None] * seq_len
#     for idx, (start, end) in enumerate(biased_ranges):
#         for pos in range(start, end + 1):
#             position_to_biased_range[pos] = idx  # Assign biased range index

#     # Build the attention matrix for the valid sequence positions
#     for i in range(seq_len):
#         biased_range_i = position_to_biased_range[i]
#         for j in range(i + 1):  # Only consider j <= i for causal mask
#             if biased_range_i is not None:
#                 # Token i is in a biased range
#                 if position_to_biased_range[j] == biased_range_i:
#                     attention_matrix[i, j] = 0.0  # Can attend within biased range
#                 else:
#                     attention_matrix[i, j] = float('-inf')  # Cannot attend outside biased range
#             else:
#                 # Token i is not in any biased range
#                 attention_matrix[i, j] = 0.0  # Can attend to all preceding tokens

#     # For positions beyond seq_len, they remain -inf (masked out)
#     return [attention_matrix.tolist()]

# def pad_attention_matrices(attention_matrices):
#     """
#     Pads a list of attention matrices to the maximum sequence length in the batch.

#     Parameters:
#     - attention_matrices: List of numpy arrays, each representing an attention matrix for a sequence.

#     Returns:
#     - List of padded attention matrices.
#     """
#     # Find the maximum sequence length in the batch
#     max_seq_len = max(len(mat) for mat in attention_matrices)
    
#     padded_matrices = []
#     for mat in attention_matrices:
#         mat = np.array(mat)
#         seq_len = mat.shape[0]
#         # Initialize a padded matrix filled with -inf
#         padded_mat = np.full((max_seq_len, max_seq_len), float('-inf'))
#         # Copy the original attention matrix into the top-left corner
#         padded_mat[:seq_len, :seq_len] = mat
#         padded_matrices.append([padded_mat.tolist()])
    
#     return padded_matrices
