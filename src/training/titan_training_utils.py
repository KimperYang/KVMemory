from dataclasses import replace
from typing import List

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

DATA_V3: List[DataComponent] = [
    DataComponent(dataset_name="text", weight=25.0),
    DataComponent(dataset_name="text_mem", weight=10.0),
    DataComponent(dataset_name="text_inst", weight=10.0),
    DataComponent(dataset_name="tulu", weight=25.0),
    DataComponent(dataset_name="sft_mem", weight=20.0),
    DataComponent(dataset_name="qa", weight=5.0),
    DataComponent(dataset_name="qa_mem", weight=5.0),
]

DATA_V4: List[DataComponent] = [
    DataComponent(dataset_name="text", weight=15.0),
    DataComponent(dataset_name="text_mem", weight=10.0),
    DataComponent(dataset_name="text_inst", weight=10.0),
    DataComponent(dataset_name="tulu", weight=50.0),
    DataComponent(dataset_name="sft_mem", weight=15.0),
    DataComponent(dataset_name="qa", weight=5.0),
    DataComponent(dataset_name="qa_mem", weight=5.0),
]

DATA_V5: List[DataComponent] = [
    DataComponent(dataset_name="text", weight=15.0),
    DataComponent(dataset_name="text_mem", weight=15.0),
    DataComponent(dataset_name="text_inst", weight=0.0),
    DataComponent(dataset_name="tulu", weight=50.0),
    DataComponent(dataset_name="sft_mem", weight=20.0),
    DataComponent(dataset_name="qa", weight=5.0),
    DataComponent(dataset_name="qa_mem", weight=5.0),
]

DATA_V6: List[DataComponent] = [
    DataComponent(dataset_name="text", weight=10.0),
    DataComponent(dataset_name="text_mem", weight=10.0),
    DataComponent(dataset_name="text_inst", weight=10.0),
    DataComponent(dataset_name="tulu", weight=25.0),
    DataComponent(dataset_name="sft_mem", weight=15.0),
    DataComponent(dataset_name="qa", weight=10.0),
    DataComponent(dataset_name="qa_mem", weight=10.0),
    DataComponent(dataset_name="xsum", weight=10.0),
]

DATASET_MAPPING = {
    "v1": DATA_V1,
    "v2": DATA_V2,
    "v3": DATA_V3,
    "v4": DATA_V4,
    "v5": DATA_V5,
    "v6": DATA_V6,
}

COMMON_CHECKPOINT_CONFIG = CheckpointConfig(
    enable_checkpoint=True,
    folder="checkpoints",
    interval_type="step",
    interval=500,
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
    eval_every_n_steps=1000,
)

bsz256_lr56_steps10k =replace(
    DEFUALT_TRAINING_RECIPE,
    batch_size=256,
)

bsz256_lr56_steps4k =replace(
    DEFUALT_TRAINING_RECIPE,
    batch_size=256,
    max_steps=4_000,
    warmup_steps=400,
    eval_every_n_steps=500,
)

bsz64_lr56_steps10k =replace(
    DEFUALT_TRAINING_RECIPE,
    batch_size=64,
)

bsz64_lr56_steps5k =replace(
    DEFUALT_TRAINING_RECIPE,
    batch_size=64,
    max_steps=5000,
    warmup_steps=500,
    eval_every_n_steps=500,
)
bsz64_lr56_steps6k =replace(
    DEFUALT_TRAINING_RECIPE,
    batch_size=64,
    max_steps=6000,
    warmup_steps=600,
    eval_every_n_steps=500,
)

bsz128_lr56_steps10k =replace(
    DEFUALT_TRAINING_RECIPE,
    batch_size=128,
)

bsz256_lr56_steps6k =replace(
    DEFUALT_TRAINING_RECIPE,
    max_steps=6000,
    warmup_steps=600,
    batch_size=256,
    eval_every_n_steps=2000,
)

TRAINING_RECIPE_MAPS = {
    "bsz32_lr25_steps10k": DEFUALT_TRAINING_RECIPE,
    "bsz256_lr56_steps10k": bsz256_lr56_steps10k
}


FULL_ACTIVATION_CHECKPOINT_CONFIG = ActivationCheckpoint(
    mode=ActivationCheckpointMode.FULL,
    selective_ac_option="op",
    # selective_ac_option="2",
)

SELECTIVE_ACTIVATION_CHECKPOINT_CONFIG = ActivationCheckpoint(
    mode=ActivationCheckpointMode.SELECTIVE,
    selective_ac_option="op",
    # selective_ac_option="2",
)


PRETRAINED_MODEL_CKPT_PATH_MAPS = {
    "meta-llama/Llama-3.2-1B-Instruct": "model_cache/Llama-3.2-1B-Instruct/model.safetensors",
}

