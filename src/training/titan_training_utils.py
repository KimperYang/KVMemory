from dataclasses import replace
from typing import List

from src.data.titan_datasets import (
    DPAwareDataLoader,
    HuggingFaceDataset,
    WeightedAggregatorDataset,
)
from src.data.titan_tokenizer import LLaMA32Tokenizer
from src.training.titan_trainer_config_utils import (
    ActivationCheckpoint,
    ActivationCheckpointMode,
    CheckpointConfig,
    DataComponent,
    TrainingRecipe,
)

DATA_V1: List[DataComponent] = [
    DataComponent(dataset_name="text", weight=20.0),
    DataComponent(dataset_name="text_mem", weight=15.0),
    DataComponent(dataset_name="text_inst", weight=15.0),
    DataComponent(dataset_name="sft", weight=25.0),
    DataComponent(dataset_name="sft_mem", weight=25.0),
]

DATA_V2: List[DataComponent] = [
    DataComponent(dataset_name="text", weight=20.0),
    DataComponent(dataset_name="text_mem", weight=15.0),
    DataComponent(dataset_name="text_inst", weight=15.0),
    DataComponent(dataset_name="sft", weight=25.0),
    DataComponent(dataset_name="sft_mem", weight=15.0),
    DataComponent(dataset_name="qa", weight=5.0),
    DataComponent(dataset_name="qa_mem", weight=5.0),
]

DATASET_MAPPING = {
    "v1": DATA_V1,
    "v2": DATA_V2,
}

COMMON_CHECKPOINT_CONFIG = CheckpointConfig(
    enable_checkpoint=True,
    folder="checkpoints",
    interval_type="step",
    interval=2000,
    model_weights_only=False,
    export_dtype="bfloat16",
    create_seed_checkpoint=False,
    async_mode="disabled",
    keep_latest_k=2,
    load_step=-1,
)

DEFUALT_TRAINING_RECIPE = TrainingRecipe(
    batch_size=32,
    lr=5e-6,
    max_steps=10_000,
    warmup_steps=1_000,
    fused=False,
    max_norm=1.0,
)

bsz256_lr56_steps10k =replace(
    DEFUALT_TRAINING_RECIPE,
    batch_size=256,
)

bsz64_lr56_steps10k =replace(
    DEFUALT_TRAINING_RECIPE,
    batch_size=64,
)



TRAINING_RECIPE_MAPS = {
    "bsz32_lr25_steps10k": DEFUALT_TRAINING_RECIPE,
    "bsz256_lr56_steps10k": bsz256_lr56_steps10k
}

DEFAULT_ACTIVATION_CHECKPOINT_CONFIG = ActivationCheckpoint(
    # mode=ActivationCheckpointMode.NONE,
    # mode=ActivationCheckpointMode.SELECTIVE,
    mode=ActivationCheckpointMode.FULL,
    selective_ac_option="op",
    # selective_ac_option="2",
)

def build_hf_data_loader(
    data_components: List[DataComponent],
    tokenizer: LLaMA32Tokenizer,
    seed: int,
    batch_size: int,
    seq_len: int,
    world_size: int,
    rank: int,
    collate_fn,
    infinite: bool = True,
):
    """Build a data loader for HuggingFace datasets."""
    all_datasets = []
    dataset_weights = []
    for data_component in data_components:
        dataset_name = data_component.dataset_name
        weight = data_component.weight
        hf_ds = HuggingFaceDataset(
            dataset_name,
            tokenizer,
            seq_len=seq_len,
            world_size=world_size,
            rank=rank,
            infinite=infinite,
        )
        all_datasets.append(hf_ds)
        dataset_weights.append(weight)

    combined_ds = WeightedAggregatorDataset(
        all_datasets,
        dataset_weights,
        seed=seed,
        infinite=infinite,
    )
    return DPAwareDataLoader(rank, combined_ds, batch_size=batch_size, collate_fn=collate_fn)


