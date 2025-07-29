import torch

def _make_bool_segment_mask(
    *,
    source_segments: torch.Tensor,
    target_segments: torch.Tensor,
) -> torch.Tensor:
    target_segments = target_segments.unsqueeze(-1)
    source_segments = source_segments.unsqueeze(-2)

    # Returning the boolean mask based on equality
    return torch.eq(source_segments, target_segments)[:, ...]

def make_segment_mask(
    *,
    source_segments: torch.Tensor,
    target_segments: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    add_causal_lm_mask: bool = True,
) -> torch.Tensor:
    """Generates attention logit biases given the segment ids.

    ... such that positions belonging to different segments cannot attend to each other.
    This function is based on the implementation of AXLearn from Apple. The original
        implementation of the `make_segment_mask` can be found in:
        https://github.com/apple/axlearn/blob/main/axlearn/common/attention_bias.py#L767

    Args:
        source_segments: An integer tensor of shape [batch, ..., source_length].
        target_segments: An integer tensor of shape [batch, ..., target_length].

    Returns:
        A float Tensor of shape [batch, 1, ..., target_length, source_length] where the
        value at [..., i, j] = 0 if target_segments[..., i] == source_segments[..., j], or -inf
        otherwise.
    """
    # min_dtype = torch.finfo(dtype).min
    min_dtype = -float("inf")
    sequence_len = source_segments.size(-1)
    batch_size = source_segments.size(0)
    if add_causal_lm_mask:
        segment_logit_bias = torch.triu(
            torch.full(
                (batch_size, sequence_len, sequence_len),
                # NEG_INF,
                min_dtype,
                dtype=dtype,
                device=source_segments.device
            ),
        diagonal=1)
    else:
        segment_logit_bias = torch.zeros(
            size=(batch_size, sequence_len, sequence_len),
            dtype=dtype,
            device=source_segments.device,
        )

    # within the same segment
    bool_mask = _make_bool_segment_mask(
        source_segments=source_segments, target_segments=target_segments
    )

    # Create masks for tokens belonging to segment 0
    # Shape [batch, ..., 1, source_length]
    target_is_zero = (target_segments == 0).unsqueeze(-1)

    # Tokens in segment 0 can be attended by any token
    zero_mask = target_is_zero

    # masks that indicates the token is a pad token
    # Shape [batch, ..., 1, source_length]
    source_invalid_mask = (source_segments == -1).unsqueeze(-2)
    target_invalid_mask = (target_segments == -1).unsqueeze(-1)
    # Combine invalid masks: pad tokens
    invalid_mask = source_invalid_mask | target_invalid_mask

    # all_masks = (~bool_mask) & (~zero_mask)
    all_masks = invalid_mask | ((~bool_mask) & (~zero_mask))
    segment_logit_bias = segment_logit_bias.masked_fill_(all_masks, min_dtype)

    # if dtype is torch.bfloat16:
    #     segment_logit_bias = segment_logit_bias.bfloat16()
    return segment_logit_bias
