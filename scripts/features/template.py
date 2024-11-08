import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaModel
checkpoint = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForCausalLM.from_pretrained(checkpoint)  # You may want to use bfloat16 and/or move to GPU here

data = [{"query": "Hello, my name is Emily. Nice to meet you.", "response": "Hello, Emily. I'm your AI companion. Can I assist you in any way?"}, {"query": "I really enjoy painting, playing piano and tea tasting. Do you have any good book recommendations for me?", "response": "Certainly, there are many books about painting techniques, piano playing and tea knowledge that I can recommend to you. Do you prefer specific technical guidance or more artistic works?"}, {"query": "I prefer artistic works, like \"The Great Gatsby\" which is full of charm.", "response": "Then I recommend \"The Little Prince\" and \"Pride and Prejudice\". Both of these books are very interesting and full of emotions."}, {"query": "Okay, thank you for your recommendations. I will take a look.", "response": "You're welcome. If you have any other needs, feel free to let me know."}]

for item in data:
    print(len(tokenizer.encode(item['query'])) + len(tokenizer.encode(item['response'])))


# tokens = tokenizer("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")['input_ids']
# print(tokens)
# print(tokenizer.convert_ids_to_tokens(tokens))
# # Add the custom token
# new_token = ["<MEM_START>","<MEM_END>", "<MEM_SUM>"]
# tokenizer.add_tokens(new_token)
# # tokenizer.pad_token = tokenizer.eos
# tokens = tokenizer("<MEM_START>Hello how are you<MEM_END><MEM_SUM>")['input_ids']
# print(tokens)
# print(tokenizer.convert_ids_to_tokens(tokens))

# messages = [
#     {
#         "role": "system",
#         "content": "You are a friendly chatbot who always responds in the style of a pirate",
#     },
#     {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
#     {"role": "assistant", "content": "Six in total"},
#     {"role": "user", "content": "How many helicopters can a human eat in one sitting?"}
#  ]
# tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")

# print(tokenizer.decode(tokenized_chat[0]))

# # print(tokenizer.decode(torch.tensor([518, 29914, 25580, 29962])))
# # print(tokenizer.decode(torch.tensor([518, 25580, 29962])))
# print(tokenizer.decode(torch.tensor([27])))