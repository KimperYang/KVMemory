import torch


def generation_cos_sin(
    position_ids: torch.LongTensor,
    dim: int = 64,
    base: int = 10000,
):
    dtype = torch.float32
    device = torch.device("cpu")
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    return cos.to(dtype=dtype), sin.to(dtype=dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
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
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def main():
    batch_size = 1
    dim = 64
    seq_len = 10

    def get_qk_v1(position_ids: torch.LongTensor):
        # batch_size, seq_len, dim
        # the first half of dim (from 0 to dim//2) corresponds to angle 0 - dim //2 * base
        # the second half of dim (from dim // 2 to dim) corresponds to angle 0 - dim //2 * base
        cos, sin = generation_cos_sin(position_ids)
        print(cos.shape)
        
        torch.random.manual_seed(2333)
        q = torch.rand(size = [batch_size, 1, seq_len, dim]).float()
        k = torch.rand(size = [batch_size, 1, seq_len, dim]).float()
        q_embed, k_embed = apply_rotary_pos_emb(q,k,cos, sin)
        return q_embed, k_embed

    # batch_size, 10
    position_ids = torch.arange(10).view(-1, 10)
    q_embed, k_embed = get_qk_v1(position_ids)
    print(q_embed.shape)
    print(k_embed[0][0][0])

    position_ids = torch.arange(10,20).view(-1, 10)
    q_embed, k_embed = get_qk_v1(position_ids)

    print(q_embed.shape)
    print(k_embed[0][0][0])

if __name__ == "__main__":
    main()

