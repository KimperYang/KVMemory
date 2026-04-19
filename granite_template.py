from transformers import AutoTokenizer

model_name = "ibm-granite/granite-4.1-8b"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
)

# print("==== Raw chat_template ====")
# print(tokenizer.chat_template)
# print()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain what a neural network is in one sentence."},
    {"role": "assistant", "content": "A neural network is a computational model inspired by the human brain, consisting of interconnected nodes (neurons) that process and learn from data."},
    {"role": "user", "content": "Explain what a neural network is in one sentence."},
    
]

rendered_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("==== Rendered prompt example ====")
print(rendered_prompt)