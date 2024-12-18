"""
Generate the 2wiki subsets with and without memory

```
python scripts/data_process/2wiki.py --max_length=4096 --validation_size=2000
```
"""
import json
from absl import app, flags
from datasets import Dataset
from transformers import AutoTokenizer

FLAGS = flags.FLAGS

def set_args():
    flags.DEFINE_integer(
        "max_length",
        default=4096,
        help="Max token length for 2Wiki",
    )
    flags.DEFINE_integer(
        "validation_size",
        default=2_000,
        help="number of samples to sample from the 2Wiki.",
    )


def main(argv):
    shards = {'train': 128, 'test': 4}

    with open("data/train.json", "r") as f:
        data = json.load(f)  # This should work if the JSON is valid

    format_list = []
    for i in range(len(data)):
        new_context = []
        for ctx in data[i]["context"]:
            new_context.append({
                "title": ctx[0],
                "sentences": " ".join(ctx[1])
            })
        format_list.append({'question': data[i]['question'], 'context': new_context, 'answer': data[i]['answer']})

    dataset = Dataset.from_list(format_list)

    total_num = len(dataset)
    first_half = dataset.select(range(0, total_num // 2))
    second_half = dataset.select(range(total_num // 2, total_num))

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    max_length = FLAGS.max_length

    def qa_filter(sample):
        # Extract "Assistant" responses and mask "User" queries
        system = "[<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question."
        system_input_ids = tokenizer(system, add_special_tokens=False).input_ids
        input_ids = system_input_ids

        if len(sample['context']) != 10:
            print("Context number not right")
            return False

        for j in range(0,10):
            title = sample['context'][j]['title']
            text = sample['context'][j]['sentences']
            # memory_list.append("<MEM_START>" + f"Document [{j+1}](Title: {title}) {text}" + "\n<MEM_END><MEM_SUM>")
            tem_id = tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False)

            input_ids += tem_id

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + sample['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = tokenizer(user, add_special_tokens=False).input_ids
        input_ids += user_id

        if len(input_ids) >= max_length:
            return False

        return True

    qa = first_half.filter(qa_filter, num_proc=96)
    qa = qa.train_test_split(test_size=FLAGS.validation_size)

    qa.save_to_disk("dataset_cache/processed/2wiki/qa", num_shards=shards, num_proc=128)
    print("qa:", qa)

    def qamem_filter(sample):
        # Extract "Assistant" responses and mask "User" queries
        system = "[<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question."
        system_input_ids = tokenizer(system, add_special_tokens=False).input_ids
        input_ids = system_input_ids

        if len(sample['context']) != 10:
            print("Context number not right")
            return False

        for j in range(0,10):
            title = sample['context'][j]['title']
            text = sample['context'][j]['sentences']
            tem_id = tokenizer("<MEM_START>" + f"Document [{j+1}](Title: {title}) {text}\n<MEM_END><MEM_SUM>", add_special_tokens=False)

            input_ids += tem_id

        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + sample['question'] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_id = tokenizer(user, add_special_tokens=False).input_ids
        input_ids += user_id

        if len(input_ids) >= max_length:
            return False

        return True

    qamem = second_half.filter(qamem_filter, num_proc=96)
    qamem = qamem.train_test_split(test_size=FLAGS.validation_size)

    qamem.save_to_disk("dataset_cache/processed/2wiki/qamem", num_shards=shards, num_proc=128)
    print("qamem:", qamem)

if __name__ == "__main__":
    set_args()
    app.run(main)
