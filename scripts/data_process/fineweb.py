from transformers import AutoTokenizer
from datasets import load_dataset
from absl import app, flags

FLAGS = flags.FLAGS

def set_args():
    flags.DEFINE_integer(
        "num_samples",
        default=10_000_000,
        help="number of samples to sample from the FineWeb.",
    )

def main(argv):
    num_samples = FLAGS.num_samples
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train"
        )

    # Your list of strings
    total_samples = len(dataset)
    print("total samples num: ", total_samples)

    # Extract the last 90,000 samples
    last_half_samples = dataset.select(range(0, total_samples))

    # Filter strings based on the number of tokens and print progress
    def filter_strings_by_token_count(strings, min_tokens=2048):
        
        # ids = tokenizer(strings['text'], add_special_tokens= False)["input_ids"]

        if(strings['token_count'] >= min_tokens):
            return True
        
        return False

    text_filtered = last_half_samples.filter(filter_strings_by_token_count)

    text_mem = text_filtered.select(range(0, len(text_filtered) // 2))
    text_inst = text_filtered.select(range(len(text_filtered) // 2, len(text_filtered)))

    random_seed = 42
    text = dataset.shuffle(seed=random_seed).select(range(0, num_samples))

    print("text:", len(text), "textmem:", len(text_mem), "text:", len(text_inst),)
    # Save the filtered dataset to the specified path
    text.save_to_disk("data/processed/fineweb/text")
    text_mem.save_to_disk("data/processed/fineweb/text_mem")
    text_inst.save_to_disk("data/processed/fineweb/text_inst")

if __name__ == "__main__":
    set_args()
    app.run(main)