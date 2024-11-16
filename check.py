import numpy as np

def construct_biased_attention_matrix(seq_len, biased_ranges, max_len):
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
    attention_matrix = np.full((max_len, max_len), float('-inf'))

    # Create a mapping from position to biased range index
    position_to_biased_range = [None] * seq_len
    for idx, (start, end) in enumerate(biased_ranges):
        for pos in range(start, end + 1):
            position_to_biased_range[pos] = idx  # Assign biased range index

    # Build the attention matrix for the valid sequence positions
    for i in range(seq_len):
        biased_range_i = position_to_biased_range[i]
        for j in range(i + 1):  # Only consider j <= i for causal mask
            if biased_range_i is not None:
                # Token i is in a biased range
                if position_to_biased_range[j] == biased_range_i:
                    attention_matrix[i, j] = 0.0  # Can attend within biased range
                else:
                    attention_matrix[i, j] = float('-inf')  # Cannot attend outside biased range
            else:
                # Token i is not in any biased range
                attention_matrix[i, j] = 0.0  # Can attend to all preceding tokens

    # For positions beyond seq_len, they remain -inf (masked out)
    return attention_matrix

# Example usage:
seq_len = 3
biased_ranges = []
max_len = 6  # Max sequence length for padding

attention_matrix = construct_biased_attention_matrix(seq_len, biased_ranges, max_len)
print("Padded Attention Matrix:")
print(attention_matrix)
