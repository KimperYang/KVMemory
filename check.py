import torch

def make_custom_segment_mask(
    *,
    first_source:  torch.Tensor,   # [B,         S]  first-level segments (1, 2, … ; 0 NOT used)
    first_target:  torch.Tensor,   # [B,         T]
    second_source: torch.Tensor,   # [B,         S]  second-level segments (only 1 or 2)
    second_target: torch.Tensor,   # [B,         T]  (not used for the rule but kept for symmetry)
    dtype: torch.dtype = torch.bfloat16,
    allow_self: bool = True        # include the current token itself (diag) in “preceding” ?
) -> torch.Tensor:
    """
    Build an attention-logit mask that enforces **both** constraints:

    1️⃣   A target token may attend *only* to **preceding** source tokens
         (causal masking).  
         - If `allow_self=True`, the token itself is also allowed.

    2️⃣   Among those preceding tokens, attention is allowed **iff**
         • they share the same *first* segment ID **OR**  
         • their *second* segment ID is 2.

    - First segments are positive (no special 0 any more).  
    - Second segments take only values 1 or 2 (value 2 is the “wild-card”).  
    - Any token whose first or second segment is -1 is treated as padding
      and cannot attend / be attended.

    Returns
    -------
    logit_bias : torch.Tensor of shape [B, T, S]  
                 0.0  → allowed  
                 -inf → masked out
    """
    NEG_INF = -float("inf")
    device   = first_source.device
    batch    = first_source.size(0)
    src_len  = first_source.size(-1)
    tgt_len  = first_target.size(-1)

    # ------------------------------------------------------------------
    # 1.  Causal mask  (only positions j <= i are allowed)
    # ------------------------------------------------------------------
    #   True  → keep        False → mask out later
    causal = torch.arange(src_len, device=device)
    causal = causal.unsqueeze(0) <= causal.unsqueeze(1)  # [S, S] lower-tri
    if not allow_self:           # strictly "preceding"
        causal.fill_diagonal_(False)
    causal = causal.expand(batch, tgt_len, src_len)      # broadcast to [B,T,S]

    # ------------------------------------------------------------------
    # 2.  First-segment agreement
    # ------------------------------------------------------------------
    fs = first_source.unsqueeze(-2)   # [B, 1, S]
    ft = first_target.unsqueeze(-1)   # [B, T, 1]
    same_first = (fs == ft)           # [B, T, S]

    # ------------------------------------------------------------------
    # 3.  Second-segment wildcard (source_second == 2)
    # ------------------------------------------------------------------
    wildcard_second = (second_source == 2).unsqueeze(-2)  # [B, 1, S] → broadcast

    # ------------------------------------------------------------------
    # 4.  Combine the logical rules
    # ------------------------------------------------------------------
    allowed = causal & ( same_first | wildcard_second )

    # ------------------------------------------------------------------
    # 5.  Padding mask  (any -1 in either segment tensor ⇒ fully invalid)
    # ------------------------------------------------------------------
    src_pad = (
        (first_source  == -1) | (second_source == -1)
    ).unsqueeze(-2)                     # [B, 1, S]
    tgt_pad = (
        (first_target  == -1) | (second_target == -1)
    ).unsqueeze(-1)                     # [B, T, 1]
    allowed &= ~(src_pad | tgt_pad)     # force to False where padded

    # ------------------------------------------------------------------
    # 6.  Convert to logit-bias tensor   (0 allowed,  -inf masked)
    # ------------------------------------------------------------------
    logit_bias = torch.full(
        (batch, tgt_len, src_len),
        NEG_INF,
        dtype=dtype,
        device=device,
    )
    logit_bias = logit_bias.masked_fill(allowed, 0.0)

    return logit_bias

B, L = 1, 6
fs = torch.tensor([[1, 1, 1, 2, 2, 3, 3]])   # first segments
ss = torch.tensor([[1, 1, 2, 1, 2, 1, 2]])   # second segments
mask = make_custom_segment_mask(
    first_source=fs,  first_target=fs,
    second_source=ss, second_target=ss,   # target second not used
    dtype=torch.float, allow_self=True
)
print(mask[0])
