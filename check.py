import torch
import transformers
from torchtune.models.convert_weights import tune_to_hf

model = torch.load("training_res/torchtune/bias_ckpt.pt", weights_only=False)
print(model['model'].keys())
import ipdb
ipdb.set_trace()
# converted_state_dict = tune_to_hf(state_dict=model, num_heads=32, num_kv_heads=8,dim=2048)

# # torch.save(model, "training_res/torchtune/pytorch_model.bin")
# print(converted_state_dict.keys())
# config = transformers.AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
# model = transformers.AutoModelForCausalLM.from_config(config)
# model.load_state_dict(converted_state_dict)

# print(model)
# # model2 = transformers.AutoModelForCausalLM.from_pretrained("training_res/torchtune")

# # print(model1.keys())
# # print()
# # print(model2.keys())
