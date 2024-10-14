import torch


def recompute_kv_cache_gradual_filtering(
    old_kv: tuple, 
    new_kv: tuple, 
    suffix_len: int, 
    initial_recomp_ratio: float, 
    decay_factor: float
) -> tuple:
    """
    Recompute the KV cache by selectively updating the old KV cache with a gradual filtering 
    scheme based on HKVD selection. The recomputation ratio decreases gradually across layers.

    Args:
    - old_kv (tuple): A tuple containing 32 layers, each layer is a tuple of (key, value),
                      where both key and value are tensors of shape 
                      [batch_size, num_heads, seq_len, head_dim].
    - new_kv (tuple): A tuple containing the newly computed key and value for 1 layer.
                      It also has the shape [batch_size, num_heads, seq_len, head_dim].
    - suffix_len (int): The number of tokens at the end of the sequence that are always recomputed.
    - initial_recomp_ratio (float): The recomputation ratio for the first layer.
    - decay_factor (float): The factor by which the recomputation ratio decreases at each layer.
                            For example, if decay_factor = 0.95, each subsequent layer will 
                            recompute 95% of the tokens selected in the previous layer.

    Returns:
    - recomputed_kv (tuple): The updated old_kv with HKVD tokens selected and recomputed gradually.
    """
    # Unpack new key and value from new_kv (since new_kv has 1 layer, we'll use this for all layers)
    new_key, new_value = new_kv

    # Make a copy of the old_kv tuple to modify it layer by layer
    updated_kv = list(old_kv)

    # Initial recomputation ratio for the first layer
    recompute_ratio = initial_recomp_ratio

    # Start by selecting all tokens for the first layer
    candidate_indices = torch.arange(old_kv[0][0].shape[2], device=new_key.device)

    # Loop through each layer of the old KV cache
    for layer_idx in range(len(old_kv)):
        # Get the old key and value for the current layer
        old_key, old_value = old_kv[layer_idx]

        # Calculate how many tokens to update based on the recompute_ratio
        num_tokens_to_update = int(len(candidate_indices) * recompute_ratio)

        # Ensure the suffix tokens are always included for recomputation
        suffix_indices = torch.arange(old_key.shape[2] - suffix_len, old_key.shape[2], device=old_key.device)

        # Compute the deviation between the old and new key/value tensors for the candidate tokens
        key_diff = torch.sum((old_key[:, :, candidate_indices, :] - new_key[:, :, candidate_indices, :]) ** 2, dim=[1, 2, 3])
        value_diff = torch.sum((old_value[:, :, candidate_indices, :] - new_value[:, :, candidate_indices, :]) ** 2, dim=[1, 2, 3])

        # Combine key and value deviation
        total_diff = key_diff + value_diff

        # Select the top-k tokens based on the deviation
        top_indices = torch.topk(total_diff, k=num_tokens_to_update).indices

        # Map the selected top indices to the candidate indices
        selected_indices = candidate_indices[top_indices]

        # Combine selected HKVD tokens with suffix tokens
        selected_indices = torch.cat([selected_indices, suffix_indices])

        # Create a copy of the old_key and old_value to update the selected tokens
        updated_key = old_key.clone()
        updated_value = old_value.clone()

        # Update the selected tokens in the old_key and old_value with the new key and value
        updated_key[:, :, selected_indices, :] = new_key[:, :, selected_indices, :]
        updated_value[:, :, selected_indices, :] = new_value[:, :, selected_indices, :]

        # Replace the old layer with the updated key and value in the old_kv
        updated_kv[layer_idx] = (updated_key, updated_value)

        # Gradually filter by setting the new candidate indices as the previously selected HKVD tokens
        candidate_indices = selected_indices[:-suffix_len]  # Exclude suffix tokens from candidates

        # Reduce the recomputation ratio for the next layer
        recompute_ratio *= decay_factor

    # Convert back to a tuple to match the original structure
    recomputed_kv = tuple(updated_kv)

    return recomputed_kv
