import torch

def rotate_half(x):
    """
    transformers.models.llama.modeling_llama.rotate_half
    Rotates half the hidden dims of the input.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(k, cos, sin, position_ids, unsqueeze_dim=1):
    """
    transformers.models.llama.modeling_llama.apply_rotary_pos_emb
    Applies Rotary Position Embedding to the query and key tensors.

    Args:
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed

def unrotate_half(x):
    """
    Inverts the rotate_half operation.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((x2, -x1), dim=-1)

def unapply_rotary_pos_emb(k_embed, cos, sin, unsqueeze_dim=1):
    """
    Reverses the Rotary Position Embedding applied to a tensor.

    Args:
        k_embed (`torch.Tensor`): The tensor after RoPE was applied.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1): 
            The same unsqueeze_dim used in apply_rotary_pos_emb.
    Returns:
        `torch.Tensor`: The original tensor before RoPE was applied.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # Reverse the RoPE operation
    k = (k_embed * cos) - (unrotate_half(k_embed) * sin)
    return k


k = torch.randn(1, 4, 15, 64)  # Example tensor
cos = torch.randn(1, 15, 64)
sin = torch.randn(1, 15, 64)

# Apply RoPE
k_rotated = apply_rotary_pos_emb(k, cos, sin, position_ids=None, unsqueeze_dim=1)

# Reverse RoPE
k_original = apply_rotary_pos_emb(k_rotated, cos, -sin, position_ids=None, unsqueeze_dim=1)

print(torch.mean(torch.abs(k-k_rotated)))
# Check if they match
print(torch.allclose(k, k_original, atol=1e-3))  # Should print True
