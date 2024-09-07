import torch

def generate_kv_with_id(model, input_ids):
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        out = model(input_ids)
        past_key_values = out.past_key_values

    return past_key_values

def append_kv(kv_list):
    num_layers = len(kv_list[0])
    concatenated_past_key_values = ()

    for layer in range(num_layers):
        keys_list = [kv[layer][0].detach() for kv in kv_list]
        values_list = [kv[layer][1].detach() for kv in kv_list]

        concatenated_keys = torch.cat(keys_list, dim=2)
        concatenated_values = torch.cat(values_list, dim=2)
        concatenated_past_key_values += ((concatenated_keys, concatenated_values),)

    return concatenated_past_key_values