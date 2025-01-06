# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
1. Download the tokenizer
python src/data/titan_download_tokenizer.py \
    --repo_id meta-llama/Llama-3.2-1B-Instruct \
    --tokenizer_path "original" \
    --local_dir data/titan_tokenizer/ \
    --hf_token=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


2. Running

```sh
LOG_RANK=${LOG_RANK:-0}
NGPU=${NGPU:-"8"}

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    titan_trainer.py --config_name block_datav1_step10k_bsz64_single_node
```

Available config names: block_datav1_step10k_bsz64_single_node, block_datav1_step10k_bsz256_4_node

3. View the tensorboard logs:
ssh -N -f -L localhost:16006:localhost:6006 satori1
tensorboard --logdir=/nobackup/users/bairu/repos/KVMemory/run_logs/block_datav1_step10k/tensorboard/20250105-1953 --port=6006
"""

import argparse
import os
import time
from dataclasses import replace
from datetime import timedelta

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import LlamaForCausalLM

from src.data.attention import construct_biased_attention_matrix
from src.data.titan_preprocessor import custom_collate_bias
from src.data.titan_tokenizer import LLaMA32Tokenizer
from src.torchtitan import utils
from src.torchtitan.logging import init_logger, logger
from src.torchtitan.optimizer import build_lr_schedulers, build_optimizers
from src.torchtitan.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from src.torchtitan.utils import device_module, device_type
from src.training.checkpointing import (
    CheckpointManager,
    TrainState,
)
from src.training.metrics import build_device_memory_monitor, build_metric_logger
from src.training.parallelism import (
    ParallelDims,
    parallelize_llama,
)
from src.training.titan_trainer_config_utils import (
    CommonConfig,
    TitanTrainerConfig,
)
from src.training.titan_training_utils import (
    COMMON_CHECKPOINT_CONFIG,
    DATASET_MAPPING,
    DEFAULT_ACTIVATION_CHECKPOINT_CONFIG,
    DEFUALT_TRAINING_RECIPE,
    bsz64_lr56_steps10k,
    bsz256_lr56_steps10k,
    build_hf_data_loader,
)

CONFIG_DICT = {
    "block_datav1_step10k_bsz64_single_node": TitanTrainerConfig(
        model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        tokenizer_path="data/titan_tokenizer/original/tokenizer.model",
        dataset_version="v1",
        seq_len=4096,
        job_dump_folder="run_logs/block_datav1_step10k",
        ckpt_config=COMMON_CHECKPOINT_CONFIG,
        training_recipe=bsz64_lr56_steps10k,
        activation_checkpoint=DEFAULT_ACTIVATION_CHECKPOINT_CONFIG,
    ),
    "block_datav1_step10k_bsz256_4_node": TitanTrainerConfig(
        model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        tokenizer_path="data/titan_tokenizer/original/tokenizer.model",
        dataset_version="v1",
        seq_len=4096,
        job_dump_folder="run_logs/block_datav1_step10k",
        ckpt_config=COMMON_CHECKPOINT_CONFIG,
        training_recipe=bsz256_lr56_steps10k,
        activation_checkpoint=DEFAULT_ACTIVATION_CHECKPOINT_CONFIG,
    ),
}

# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(config_name: str):
    init_logger()
    task_config = CONFIG_DICT[config_name]
    common_cfg = CommonConfig()

    job_dump_folder = task_config.job_dump_folder
    log_freq = 10
    train_timeout_seconds = 100


    logger.info(f"Starting job: {config_name}")

    logger.info(f"Running with args:\n{task_config}")

    # used for colorful printing
    color = utils.Color

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=common_cfg.gc_freq)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        # dp_shard=1,
        # dp_replicate=8,
        dp_shard=-1,
        dp_replicate=1,
        cp=1,
        tp=1,
        pp=1,
        world_size=world_size,
        enable_loss_parallel=False,
    )
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    device_module.set_device(device)
    utils.init_distributed(
        job_dump_folder=job_dump_folder
    )
    # initialize device memory monitor and get peak flops for MFU calculation
    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0
    local_batch_size = task_config.training_recipe.batch_size // dp_degree


    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]

    # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
    utils.set_determinism(
        world_mesh, device, common_cfg.seed, common_cfg.deterministic
    )
    model_name = task_config.model_name_or_path
    tokenizer_path = task_config.tokenizer_path

    # build tokenizer
    tokenizer = LLaMA32Tokenizer(tokenizer_path)
    # build dataloader
    data_components = DATASET_MAPPING[task_config.dataset_version]
    data_loader = build_hf_data_loader(
        data_components,
        tokenizer,
        seed=common_cfg.seed,
        batch_size=local_batch_size,
        seq_len=task_config.seq_len,
        world_size=dp_degree,
        rank=dp_rank,
        infinite=True,
        collate_fn=custom_collate_bias,
    )

    # build model (using meta init)
    logger.info(f"Building {model_name}...")
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_flash_attention_2=False,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16,
        use_cache=True,
        device_map=None,
    )
    # must add this line of code
    model.gradient_checkpointing_enable()

    # log model size
    model_param_count = utils.get_num_params(model)
    logger.info(
        f"{color.blue}Model {model_name}"
        f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
    )

    # loss function to be shared by Pipeline Parallel and SPMD training
    # def loss_fn(pred, labels):
    #     return torch.nn.functional.cross_entropy(
    #         pred.flatten(0, 1).float(), labels.flatten(0, 1)
    #     )

    # init_device = device_type
    # buffer_device = None

    model = model.to(device_type)
    # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
    parallelize_llama(
        model,
        world_mesh,
        parallel_dims,
        activation_checkpoint=task_config.activation_checkpoint,
    )
    # model.to_empty(device=init_device)
    # with torch.no_grad():
    #     model.init_weights(buffer_device=buffer_device)
    model.train()

    model_parts = [model]

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    # build optimizer after applying parallelisms to the model
    optimizers = build_optimizers(
        model_parts,
        lr=task_config.training_recipe.lr,
        fused=task_config.training_recipe.fused,
    )
    lr_schedulers = build_lr_schedulers(
        optimizers.optimizers,
        steps=task_config.training_recipe.max_steps,
        warmup_steps=task_config.training_recipe.warmup_steps,
    )

    train_state = TrainState()

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=data_loader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        ckpt_config=task_config.ckpt_config,
    )

    if task_config.ckpt_config.create_seed_checkpoint:
        assert (
            world_size == 1
        ), "Must create seed-checkpoint using one gpu, to disable sharding"
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint.load(step=task_config.ckpt_config.load_step)
    metric_logger = build_metric_logger(
        parallel_dims,
        dump_folder=job_dump_folder,
    )

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0:
        for idx, step in enumerate(train_state.log_steps):
            metrics = {
                "loss_metrics/global_avg_loss": train_state.global_avg_losses[idx],
                "loss_metrics/global_max_loss": train_state.global_max_losses[idx],
            }
            metric_logger.log(metrics, step=step)

    data_iterator = iter(data_loader)

    # variables used to keep info for metrics logging
    losses_since_last_log = []
    ntokens_since_last_log = 0
    data_loading_times = []
    time_last_log = time.perf_counter()
    device_memory_monitor.reset_peak_stats()

    checkpoint.reset()

    # train loop
    logger.info(
        f"Training starts at step {train_state.step + 1}, "
        f"with local batch size {local_batch_size}, "
        f"global batch size {task_config.training_recipe.batch_size}, "
        f"sequence length {task_config.seq_len}, "
        f"total steps {task_config.training_recipe.max_steps} "
        f"(warmup {task_config.training_recipe.warmup_steps})"
    )
    with maybe_enable_profiling(
        # TODO: check if this works
        enable_profiling=True,
        job_dump_folder=job_dump_folder,
        global_step=train_state.step
    ) as torch_profiler, maybe_enable_memory_snapshot(
        global_step=train_state.step
    ) as memory_profiler:
        while train_state.step < task_config.training_recipe.max_steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            # get batch
            data_load_start = time.perf_counter()
            batch = next(data_iterator)
            # input_ids, labels = batch
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_matrices = []
            max_length = max(batch['input_length'])
            for idx in range(len(batch['input_ids'])):
                mem_num = batch['mem_num'][idx]
                if mem_num == 0:
                    biased_ranges = None
                else:
                    biased_ranges = batch['biased_index'][idx][:mem_num]
                attention_matrices.append(
                    construct_biased_attention_matrix(
                        batch['input_length'][idx],
                        biased_ranges,
                        max_length,
                        batch['input_ids'].device
                    ).unsqueeze(0)
                )


            ntokens_since_last_log += labels.numel()
            data_loading_times.append(time.perf_counter() - data_load_start)

            input_ids = input_ids.to(device_type)
            labels = labels.to(device_type)
            attention_mask = torch.stack(attention_matrices).to(device_type)
            # print(input_ids)
            # print(labels)
            # print(attention_mask)
            optimizers.zero_grad()

            # Non-PP forward / backward
            # pred = model(input_ids)
            # loss = loss_fn(pred, labels)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                # output_hidden_states=True,
            )
            loss = outputs.loss
            # pred.shape=(bs, seq_len, vocab_size)
            # need to free to before bwd to avoid peaking memory
            # del pred
            loss.backward()

            # clip gradients
            grad_norm = utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
                task_config.training_recipe.max_norm,
                foreach=True,
                pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            )

            # optimizer step
            checkpoint.maybe_wait_for_staging()
            optimizers.step()
            lr_schedulers.step()

            losses_since_last_log.append(loss)

            # log metrics
            if (
                train_state.step == 1
                or train_state.step % log_freq == 0
            ):
                losses = [loss.item() for loss in losses_since_last_log]
                avg_loss, max_loss = sum(losses) / len(losses), max(losses)
                if (
                    parallel_dims.dp_replicate_enabled
                    or parallel_dims.dp_shard_enabled
                    or parallel_dims.cp_enabled
                ):
                    global_avg_loss, global_max_loss = (
                        utils.dist_mean(avg_loss, world_mesh["dp_cp"]),
                        utils.dist_max(max_loss, world_mesh["dp_cp"]),
                    )
                    grad_norm = grad_norm.item()
                    global_grad_norm = utils.dist_mean(grad_norm, world_mesh["dp_cp"])
                else:
                    global_avg_loss, global_max_loss = avg_loss, max_loss

                # update train state
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                time_delta = time.perf_counter() - time_last_log

                # tokens per second per device, abbreviated as tps
                tps = ntokens_since_last_log / (
                    time_delta * parallel_dims.non_data_parallel_size
                )
                # model FLOPS utilization
                # For its definition and calculation, please refer to the PaLM paper:
                # https://arxiv.org/abs/2204.02311

                time_end_to_end = time_delta / log_freq
                time_data_loading = sum(data_loading_times) / len(data_loading_times)
                time_data_loading_pct = 100 * sum(data_loading_times) / time_delta

                device_mem_stats = device_memory_monitor.get_peak_stats()

                curr_lr = optimizers.optimizers[0].param_groups[0]["lr"]

                metrics = {
                    "loss_metrics/global_avg_loss": global_avg_loss,
                    "loss_metrics/global_max_loss": global_max_loss,
                    "throughput(tps)": tps,
                    "time_metrics/end_to_end(s)": time_end_to_end,
                    "time_metrics/data_loading(s)": time_data_loading,
                    "time_metrics/data_loading(%)": time_data_loading_pct,
                    "memory/max_active(GiB)": device_mem_stats.max_active_gib,
                    "memory/max_active(%)": device_mem_stats.max_active_pct,
                    "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
                    "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
                    "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
                    "memory/num_ooms": device_mem_stats.num_ooms,
                    "training/lr": curr_lr,
                    "training/grad_norm": global_grad_norm,
                }
                metric_logger.log(metrics, step=train_state.step)

                logger.info(
                    f"{color.cyan}step: {train_state.step:2}  "
                    f"{color.green}loss: {global_avg_loss:7.4f}  "
                    f"{color.yellow}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
                    f"({device_mem_stats.max_reserved_pct:.2f}%)  "
                    f"{color.blue}tps: {round(tps):,}  "
                    f"{color.magenta}%{color.reset}"
                )

                losses_since_last_log.clear()
                ntokens_since_last_log = 0
                data_loading_times.clear()
                time_last_log = time.perf_counter()
                device_memory_monitor.reset_peak_stats()

            checkpoint.save(
                train_state.step, force=(train_state.step == task_config.training_recipe.max_steps)
            )

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                utils.set_pg_timeouts(
                    timeout=timedelta(seconds=train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metric_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        default="block_datav1_step10k_bsz64_single_node",
        type=str,
    )
    args = parser.parse_args()
    config_name = args.config_name
    main(config_name)
    torch.distributed.destroy_process_group()

