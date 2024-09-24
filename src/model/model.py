import torch
import torch.nn as nn
from src.utils.cache import generate_kv_with_id, concat_kv, append_kv

class CustomLlamaModel(nn.Module):
    def __init__(self, base_model, head_dim, num_heads, num_layers):
        super(CustomLlamaModel, self).__init__()
        self.base_model = base_model

        self.trainable_key_caches = nn.ParameterList([
            nn.Parameter(nn.init.normal_(torch.empty(1, num_heads, 1, head_dim, dtype=torch.bfloat16))) for _ in range(num_layers)
        ])
        
        self.trainable_value_caches = nn.ParameterList([
            nn.Parameter(nn.init.normal_(torch.empty(1, num_heads, 1, head_dim, dtype=torch.bfloat16))) for _ in range(num_layers)
        ])

        self.device = self.base_model.device

    def get_device(self):

        return next(self.base_model.parameters()).device

    def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None, use_cache=True):
        # print("1,",input_ids.device)
        # input_ids.to(self.get_device())
        # print("2,",input_ids.device)
        # print(self.get_device())
        # print(self.base_model.device)
        if past_key_values is not None:
            trainable_kv = tuple(zip(self.trainable_key_caches, self.trainable_value_caches))

            kv_s = generate_kv_with_id(self.base_model, torch.tensor([[1]]))

            trainable_kv_batch = append_kv([trainable_kv] * past_key_values[0][0].size(0),0)
            past_key_values_connect = append_kv([past_key_values, trainable_kv_batch],2)

            num_memory = int((501 - 1) / 50)
            past_key_values_batch = concat_kv(past_key_values_connect, num_memory)
            
            kv_s_concat = append_kv([kv_s] * past_key_values_batch[0][0].size(0),0)
            past_key_values_batch = append_kv([kv_s_concat, past_key_values_batch],2)

            # Pass the concatenated KV cache to the LLaMA model
            output = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels = labels, past_key_values=past_key_values_batch, use_cache=use_cache)
        
        else:
            output = self.base_model(input_ids=input_ids)

        return output