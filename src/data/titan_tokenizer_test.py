import datasets
from absl.testing import parameterized
from transformers import AutoTokenizer

from src.data.input_preprocessor import bias_attention_preprocessor as HF_PROCESSOR
from src.data.titan_preprocessor import bias_attention_preprocessor as LLAMA_PROCESSOR
from src.data.titan_tokenizer import LLaMA32Tokenizer

LLAMA_tokenizer_path="data/titan_tokenizer/original/tokenizer.model"
DATASET_NAME_TO_PATH = {
    "pretrain_process": "dataset_cache/processed/fineweb/text",
    "tulu_process": "dataset_cache/processed/tulu/sft",
    "qa_process": "dataset_cache/processed/block_qa/qa",
    "qa_mem_process": "dataset_cache/processed/block_qa/qa_mem"
}
DATASET_NAME_TO_PROCESSOR = {
    "pretrain_process": "process_text",
    "tulu_process": "process_tulu",
    "qa_process": "process_qa",
    "qa_mem_process": "process_qamem"
}


class InputTokenizationTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llama_tokenizer = LLaMA32Tokenizer(LLAMA_tokenizer_path)
        cls.hf_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        cls.llama_preprocessor = LLAMA_PROCESSOR(cls.llama_tokenizer, max_len=4096)
        cls.hf_preprocessor = HF_PROCESSOR(cls.hf_tokenizer, max_len=4096)

    def compare_tokenized_outputs(self, example):
        llama_tokenized = self.llama_tokenizer(example, add_special_tokens=False)
        hf_tokenized = self.hf_tokenizer(example, add_special_tokens=False)
        self.assertEqual(
            llama_tokenized["input_ids"], hf_tokenized["input_ids"],
            f"Tokenized IDs differ for input: {example}"
        )

    def compare_processed_datasets(self, data_path, llama_fn, hf_fn):
        full_ds = datasets.load_from_disk(data_path)["train"]
        full_ds.cleanup_cache_files()
        ds = full_ds.select(range(10))
        del full_ds

        llama_ds = ds.map(llama_fn, batched=False, load_from_cache_file=False)
        hf_ds = ds.map(hf_fn, batched=False, load_from_cache_file=False)

        for idx in range(len(llama_ds)):
            llama_example = llama_ds[idx]
            hf_example = hf_ds[idx]

            self.assertDictEqual(
                llama_example, hf_example,
                f"Mismatch found in preprocessed examples at index {idx}"
            )

    def test_tokenization(self):
        data_path = DATASET_NAME_TO_PATH["pretrain_process"]
        ds = datasets.load_from_disk(data_path)["train"]
        test_examples = ds["text"][:10]
        for example in test_examples:
            self.compare_tokenized_outputs(example)

    @parameterized.named_parameters(
        ("pretrain_process", "pretrain_process"),
        ("tulu_process", "tulu_process"),
        ("qa_process", "qa_process"),
        ("qa_mem_process", "qa_mem_process"),
    )
    def test_processing_pipeline(self, dataset_name):
        # Retrieve data path and processor function dynamically
        data_path = DATASET_NAME_TO_PATH[dataset_name]
        process_fn_name = DATASET_NAME_TO_PROCESSOR[dataset_name]

        # Get preprocessing functions
        llama_fn = getattr(self.llama_preprocessor, process_fn_name)
        hf_fn = getattr(self.hf_preprocessor, process_fn_name)

        # Compare processed datasets
        self.compare_processed_datasets(data_path, llama_fn, hf_fn)

