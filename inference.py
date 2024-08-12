import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DynamicCache
import json

max_memory = {}

def generate_kv(prompt):

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map="auto")
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # out = model(**inputs, use_cache=True)
    print('device',model.device)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    print(input_ids)
    input_ids = input_ids.to(model.device)
    out = model(input_ids)
    past_key_values = out.past_key_values

    #filter <s>
    filtered_past_key_values = ()

    for past_keys, past_values in past_key_values:

        filtered_keys = past_keys[:, :, 1:, :] 
        filtered_values = past_values[:, :, 1:, :] 
        filtered_past_key_values = filtered_past_key_values + ((filtered_keys, filtered_values),)

    print(filtered_past_key_values[0][0].size())
    # print(filtered_past_key_values.get_seq_length())
    return filtered_past_key_values

def append_kv(kv_list):
    if not kv_list:
        raise ValueError("kv_list is empty. It must contain at least one past_key_values list.")

    num_layers = len(kv_list[0])

    concatenated_past_key_values = ()

    for layer in range(num_layers):
        
        keys_list = [kv[layer][0] for kv in kv_list]
        values_list = [kv[layer][1] for kv in kv_list]

        # Concatenate keys and values along the sequence length dimension
        concatenated_keys = torch.cat(keys_list, dim=2)
        concatenated_values = torch.cat(values_list, dim=2) 

        concatenated_past_key_values = concatenated_past_key_values + ((concatenated_keys, concatenated_values),)
    # keys, values = concatenated_past_key_values[0], concatenated_past_key_values[1]
    # torch.save(keys, "keys.pt")
    # torch.save(values, "values.pt")
    return concatenated_past_key_values

def inference_with_kv(prompt, past_key_values, model_name="meta-llama/Llama-2-7b-chat-hf", max_length=800, num_return_sequences=1):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    # past_key_values = (
    #     (keys.to(model.device), values.to(model.device))
    #     for keys, values in past_key_values
    # )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # input_ids = input_ids[:, 1:]
    print(input_ids)
    # for token_id in input_ids:
    #     token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    #     print(f"Token ID: {token_id}, Token: '{token}'")
    input_ids = input_ids.to(model.device)
    # inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
    model.eval()

    # past_key_values = DynamicCache()
    # past_key_values.key_cache = torch.load("keys.pt")
    # past_key_values.value_cache =values = torch.load("values.pt")
    # print("p2",past_key_values.get_seq_length())
    with torch.no_grad():

        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            past_key_values=past_key_values,
            use_cache=True
        )
    # print(outputs)
    generated_sequences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return generated_sequences

def count_tokens(input_text):

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')

    tokens = tokenizer.encode(input_text, add_special_tokens=True)

    num_tokens = len(tokens)
    for token_id in tokens:
        token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        print(f"Token ID: {token_id}, Token: '{token}'")
    print(f"Number of tokens including special tokens: {num_tokens}")

# memory_list = ["hello\n", "how are you\n", "what is your name\n"]
memory_list = ["what is your name\n"]
# template = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible.\n<</SYS>>\n\n"

template = "[INST] <<SYS>>\nYou're an assistant who answer the question with the knowledge provided in the prompt\n<</SYS>>\n\n"
memory_list = [
"Can brain cells move? By movement I mean long distance migration (preferably within the brain only).",
"The question is relatively broad and one should take into account that the brain not only consists of neurons, but also glial cells (supportive cells) and pre-mitotic neuronal stem cells.",
"Furthermore, as critical fellow-scientists have indicated, developmental stage is very important, as the developing embryonic brain is very different from the adult brain.\n",
"However, after sifting through various publications, the answer to the question is actually remarkably simple: Yes, brain cells migrate.\n",
"In the adult brain glial cells migrate in the brain (Klämbt, 2009). Glial cells are involved in a myriad of functions, but a notable example of migrating glial cells are the oligodendrocytes that migrate relative long distances to find their target axons onto which they wrap themselves to form the insulating myelin sheath (Tsai and Miller, 2002).\n",
"Neuronal stem cells migrate over long distances in response to injury (Imitola et al., 2004) and they migrate from specific stem-cell locations (e.g., hippocampus and subventricular zone) to other regions (Clarke, 2003).\n",
"Post-mitotic, but non-differentiated neurons have been shown to migrate in the adult brain in fish (Scott et al., 2012), and in mammals and non-human primates as well (Sawada et al., 2011).\n",
"Not surprisingly, glial cells, stem cells and neurons also migrate during embryonic development. Most notably, post-mitotic neurons destined to fulfill peripheral functions have to migrate over relatively long distances from the neural crest to their target locations (Neuroscience, 2nd ed, Neuronal Migration).\n",
"Question: which kinds of brain cells can move?"
]

# template = "[INST] <<SYS>>\nYou're an assistant who answer the question with the multiple turns of history conversations\n<</SYS>>\n\n"
# memory_list = [
#     "User: My name is Ben Luttrel. Please remember my name.\nAssistant: Sure! I will remember it.\n",
#     "User: Can you tell me about your capabilities?\nAssistant: I can assist with a variety of tasks such as answering questions, providing information, generating text, and helping with problem-solving.\n",
#     "User: I need help with a coding problem. I'm working with Python and getting an error with a list comprehension.\nAssistant: Sure, I’d be happy to help. Could you share the code and the error message you’re encountering?\n",
#     "User: Here’s my code: `numbers = [x for x in range(10) if x % 2 = 0]`. The error says \"SyntaxError: invalid syntax\".\nAssistant: It looks like there's a typo. You should use `==` for comparison, not `=`. Try this: `numbers = [x for x in range(10) if x % 2 == 0]`.\n",
#     "User: Got it, thanks! Can you also help me with some statistics questions?\nAssistant: Of course! What do you need help with in statistics?\n",
#     "User: I’m working on a regression analysis and need to understand the concept of multicollinearity.\nAssistant: Multicollinearity occurs when two or more predictor variables in a regression model are highly correlated, making it difficult to determine the individual effect of each predictor. It can lead to unreliable estimates of regression coefficients.\n",
#     "User: What are some ways to detect multicollinearity?\nAssistant: You can detect multicollinearity using several methods, such as calculating the Variance Inflation Factor (VIF) for each predictor, examining the correlation matrix of predictors, or using condition indices from a condition number analysis.\n",
#     "User: Could you explain how to interpret the VIF values?\nAssistant: VIF values indicate how much the variance of a regression coefficient is inflated due to multicollinearity. A VIF value of 1 suggests no correlation, while a VIF above 5 or 10 may indicate problematic multicollinearity.\n",
#     "User: I’ve also heard about Principal Component Analysis (PCA). How is it related to multicollinearity?\nAssistant: PCA can help address multicollinearity by transforming the predictors into a set of uncorrelated components. These components are linear combinations of the original variables, and they can be used in place of the original variables in the regression model to reduce multicollinearity.\n",
#     "User: That’s helpful. Thanks for the information! Is there anything else I should know about PCA?\nAssistant: PCA is a powerful technique for dimensionality reduction, but it’s important to remember that it can make the interpretation of the resulting components more complex. It’s also a good idea to standardize your data before applying PCA if the variables are on different scales.\n"
#     "User: What is my name?"
# ]

start_token = "<s>"
end_token = "[/INST]"
memory_list.insert(0, template)
# memory_list.insert(0, start_token)
memory_list.append(end_token)

kv_list = []
seq = ""

# for i in range(len(memory_list)):
#     memory_list[i] = memory_list[i]

for st in memory_list:
    kv_list.append(generate_kv(st))
    seq = seq + st

appended_kv = append_kv(kv_list)


# print(appended_kv[0][0].size())
# print(len(appended_kv))
# print(len(appended_kv[0]))
# print(seq)
# count_tokens(seq)


# seq_cache = generate_kv(seq)
# print(inference_with_kv(seq, seq_cache))
print(inference_with_kv(seq, appended_kv))



# start_kv = generate_kv("<s>")
# template_kv = generate_kv("[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible.\n<</SYS>>\n\n [/INST]")
# dummy_kv1 = generate_kv("hello") 
# dummy_kv2 = generate_kv("how are you")
# dummy_kv3 = generate_kv("and how old are you")


# appended_kv = append_kv([template_kv, start_kv, dummy_kv1, dummy_kv2, dummy_kv3])
# print(appended_kv[0][0].size())
# print(inference_with_kv("[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible.\n<</SYS>>\n\n [/INST] hello ha ha ha how are you and how old are you", appended_kv))
