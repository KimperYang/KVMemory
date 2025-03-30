from transformers import AutoTokenizer
from transformers import PreTrainedModel
# Define model names
model_8b = "meta-llama/Meta-Llama-3-8B-Instruct"
model_1b = "meta-llama/Llama-3.2-1B-Instruct"

# Load tokenizers
tokenizer_8b = AutoTokenizer.from_pretrained(model_8b)
tokenizer_1b = AutoTokenizer.from_pretrained(model_1b)

# Example conversation
conversation = [
    {"role": "system", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What about Germany?"}
]

# Apply chat templates
rendered_8b = tokenizer_8b.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
rendered_1b = tokenizer_1b.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

# Compare and print results
if rendered_8b == rendered_1b:
    print("✅ The rendered chat templates are the same.")
else:
    print("❌ The rendered chat templates are different.\n")
    print("---- 8B Template Output ----\n")
    print(rendered_8b)
    print("\n---- 1B Template Output ----\n")
    print(rendered_1b)
