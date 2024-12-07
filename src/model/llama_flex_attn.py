from transformers import LlamaForCausalLM


class FlexAttnLlama(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
