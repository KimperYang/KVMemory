import torch
from transformers import AutoTokenizer
from datasets import load_dataset
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
data = load_dataset("nvidia/Daring-Anteater")

def process_conversation(conversation):
    # Extract "Assistant" responses and mask "User" queries
    system = "[INST] <<SYS>>\nYou're an assistant who answer the question with the knowledge provided in the prompt\n<</SYS>>\n\n"
    text = system
    system_tokenized = tokenizer(text, return_tensors="pt")
    system_input_ids = system_tokenized.input_ids
    system_attention_msk = system_tokenized.attention_mask

    mask = []
    mask.extend([0] * system_input_ids.size(1))
    input_ids_list = [system_input_ids]
    attention_mask_list = [system_attention_msk]

    for i in range(len(conversation)):
        if conversation[i]["from"] == "User":
            if i==0:
                t = conversation[i]["value"] + " [/INST] "
            else:
                t = " </s><s>[INST] " + conversation[i]["value"]  + " [/INST] " 
            text += t
            tokenized = tokenizer(t, return_tensors="pt")

            input_ids = tokenized.input_ids[:, 1:]
            if len(mask) + input_ids.size(1) > 2048: 
                break
            attention_msk = tokenized.attention_mask[:, 1:]

            mask.extend([0] * input_ids.size(1))
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_msk)

        elif conversation[i]["from"] == "Assistant":
            t = conversation[i]["value"]
            text += t
            tokenized = tokenizer(t, return_tensors="pt")

            input_ids = tokenized.input_ids[:, 1:]
            if len(mask) + input_ids.size(1) > 2048: 
                input_ids = input_ids[:, :2048 - len(mask)]
                # break
            attention_msk = tokenized.attention_mask[:, 1:]

            mask.extend([1] * input_ids.size(1))
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_msk)

        # text += turn["value"] + " " 
        # tokenized = self.tokenizer(text, return_tensors="pt")
        # input_ids = tokenized.input_ids
        # attention_msk = tokenized.attention_mask

        # if len(mask) + input_ids.size(1) > self.max_length:
        #     break

        # if turn["from"] == "User":
        #     mask.extend([0] * input_ids.size(1))
        # elif turn["from"] == "Assistant":
        #     mask.extend([1] * input_ids.size(1)) 

        # input_ids_list.append(input_ids)
        # attention_mask_list.append(attention_msk)
    
    input_ids = torch.cat(input_ids_list, dim=1)
    attention_mask = torch.cat(attention_mask_list, dim=1)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'dataset_id': text,
        'loss_mask': torch.tensor(mask).unsqueeze(0)  # Mask to ignore User's tokens during loss computation
    }

# sample = [ { "from": "User", "value": "The digits 1, 3 and 5 are each used once to form each of the possible three-digit positive integers. The three-digit integers are listed from greatest to least. Which integer is listed fifth?" }, 
#            { "from": "Assistant", "value": "I need to find the possible three-digit integers that can be formed using the digits 1, 3 and 5, without repeating any of them. I can start by finding the greatest such integer, which is 531, since I want to use the largest digit in the hundreds place, the next largest in the tens place, and the smallest in the units place. To find the next greatest integer, I can keep the hundreds digit as 5, but switch the tens and units digits, so I get 513. To find the next one, I can change the hundreds digit to 3, and use the largest remaining digit, 5, in the tens place, and the smallest remaining digit, 1, in the units place, so I get 351. To find the next one, I can keep the hundreds digit as 3, but switch the tens and units digits, so I get 315. To find the fifth greatest integer, I can change the hundreds digit to 1, and use the largest remaining digit, 5, in the tens place, and the next largest remaining digit, 3, in the units place, so I get 153. Therefore, the integer that is listed fifth is 153. " }, 
#            { "from": "User", "value": "California experiences heavy earthquake activity due to: \nA: errostion\nB: techtonics\nC: volcanic activity\nD: fire\na tectonic plate moves along a fault line" }, 
#            { "from": "Assistant", "value": "The answer is B: techtonics" } ]

sample = [ { "from": "User", "value": "Hello" }, 
           { "from": "Assistant", "value": "Hi, how are you?" }, 
           { "from": "User", "value": "Good" }, 
           { "from": "Assistant", "value": "Good" } ]

# print(data['train']['conversations'][131][0]['from'])
print(data['train']['conversations'][131][1]['value'])
print(process_conversation(data['train']['conversations'][131])['input_ids'])
print(process_conversation(data['train']['conversations'][131])['loss_mask'])

# tokenized = tokenizer(process_conversation(sample)['dataset_id'], return_tensors="pt")

# print(tokenized['input_ids'])