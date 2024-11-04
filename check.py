# from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
# from peft import PeftModel, PeftConfig
# from safetensors import safe_open
# import torch

# def load_lora_weights(adapter_path):
#     tensors = {}
#     with safe_open(adapter_path + '/adapter_model.safetensors', framework="pt", device="cpu") as f:
#         for key in f.keys():
#             tensors[key] = f.get_tensor(key)
#     return tensors

# def check_embedding_layer_length(model_name):
#     # Load the tokenizer and the model
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

#     embed_before_resize = model.state_dict()['model.embed_tokens.weight']
#     # model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
#     # ref_model.resize_token_embeddings(len(tokenizer))
#     # model = AutoModel.from_pretrained(model_name)

#     # print(torch.sum(ref_model.get_input_embeddings().weight - model.get_input_embeddings().weight))
#     # import pdb
#     # pdb.set_trace()
#     peft_config_path = model_name  # Path to the directory where LoRA weights are stored
#     vocab_size = len(tokenizer)
#     model.resize_token_embeddings(vocab_size)
#     embed_after_resize = model.state_dict()['model.embed_tokens.weight']
#     print('resize_diff: ', torch.sum(embed_before_resize - embed_after_resize[:128256]))
#     print(embed_before_resize.shape, embed_after_resize.shape)
#     old_embedding = model.state_dict()['model.embed_tokens.weight']

#     model = PeftModel.from_pretrained(model, peft_config_path)

#     # print(load_lora_weights(model_name)['base_model.model.model.embed_tokens.modules_to_save.weight'][128258])
#     # print(load_lora_weights(model_name)['base_model.model.model.embed_tokens.original_module.weight'][128258])
#     # print(load_lora_weights(model_name)['base_model.model.model.embed_tokens.weight'][128258])
#     lora_weights = load_lora_weights(model_name)
#     # print(lora_weights.keys())
#     print('lora_diff: ', torch.sum(lora_weights['base_model.model.model.embed_tokens.modules_to_save.weight'][:128256] - lora_weights['base_model.model.model.embed_tokens.original_module.weight'][:128256]))
    
#     # new_embedding = model.base_model.model.embed_tokens.modules_to_save.default.weight
#     new_embeding = model.state_dict()['base_model.model.model.embed_tokens.modules_to_save.default.weight'][:128256]
#     print('model_new_old_diff: ', torch.sum(new_embeding[:128256] - old_embedding[:128256]))
#     print('model_lora_diff: ',torch.sum(lora_weights['base_model.model.model.embed_tokens.modules_to_save.weight'][:128256] - new_embeding[:128256]))
#     print('model_lora_diff: ',torch.sum(lora_weights['base_model.model.model.embed_tokens.original_module.weight'][:128256] - old_embedding[:128256]))
# # Example usage:
# model_name = "/mnt/data/jingbo/kv_dump_combine_mix5_5000steps"  # Replace with your desired model
# # model_name = "/mnt/data/jingbo/kv_dump_combine_special2"  # Replace with your desired model
# check_embedding_layer_length(model_name)

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

xsum = load_from_disk('/mnt/data2/jingbo/kvmemory/data/maxlen4096/xsum_min5paragraphs')
print(xsum[10])
# cnn = load_dataset('EdinburghNLP/xsum')
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# for i in range(len(cnn['train']['document'])):
#     print(len(tokenizer(cnn['train']['document'][i])['input_ids']))

