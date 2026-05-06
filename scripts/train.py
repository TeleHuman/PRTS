import os
import gc
gc.disable()

import sys
import logging
from pathlib import Path
import time
import math

import torch
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list
import torch.distributed as dist

try:
    import deepspeed.runtime.utils as ds_utils
    import deepspeed.runtime.zero.stage_1_and_2 as zero2_core
    
    def _fast_count_used_parameters(model_or_params, *args, **kwargs):
        if hasattr(model_or_params, "parameters"):
            return sum(1 for p in model_or_params.parameters() if getattr(p, "requires_grad", False))
        try:
            return sum(1 for p in model_or_params if getattr(p, "requires_grad", False))
        except TypeError:
            return 0

    ds_utils.count_used_parameters_in_backward = _fast_count_used_parameters
    if hasattr(zero2_core, "count_used_parameters_in_backward"):
        zero2_core.count_used_parameters_in_backward = _fast_count_used_parameters
        
    print("✅ [Memory patch injected] DeepSpeed timing hook has been fully removed!")
except ImportError:
    pass

# TODO (zy): currently we do not support LORA fine-tuning
# try:
#     from peft import LoraConfig, get_peft_model
# except ImportError:
#     pass
from transformers import TrainerCallback
from transformers import HfArgumentParser

from prts.models.modeling_prts_qwen3_vl import PRTS_Qwen3VL
from prts.models.configuration_prts_qwen3_vl import PRTS_FlowMatchingConfig_Qwen3VL
from prts.models.processing_prts_qwen3_vl import PRTS_Qwen3VLProcessor
from prts.models.processing_action_tokenizer import UniversalActionProcessor
from prts.training.config import TrainConfig
from prts.training.train_utils import (
    configure_llm_qwen3_vl,
    configure_vision_tower,
    safe_save_model_for_hf_trainer,
    replace_qwen2_vl_attention_class,
    LOG_FORMATTER,
)
from prts.data.dataset import (
    make_dataset,
    pack_dataset,
    PackedDataset,
)
from prts.training.trainer import PRTSTrainer

logger = get_logger(__name__, log_level="INFO")
if not logger.logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(LOG_FORMATTER)
    logger.logger.addHandler(handler)


class MemoryMonitorCallback(TrainerCallback):
    def __init__(self, log_interval=1):
        self.log_interval = log_interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_interval == 0:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(device) / 1024**2
                reserved = torch.cuda.memory_reserved(device) / 1024**2
                max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2

import ctypes
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

class EnhancedGCCallback(TrainerCallback):
    """
    工业级分布式受控内存清理机制 (Enterprise Distributed Amortized GC)
    
    特点：
    1. 进程同步：使用 barrier 确保分布式训练不触发 NCCL Timeout。
    2. 深度清理：同时清理 Python GC、底层 glibc 碎片、CUDA 显存缓存、以及 IPC 通信残留。
    3. 精准监控：实时打印清理前后的物理 RAM 和 VRAM 差值。
    """
    def __init__(self, full_gc_interval=1000):
        self.full_gc_interval = full_gc_interval
        self.process = psutil.Process(os.getpid()) if HAS_PSUTIL else None
        
        try:
            self.libc = ctypes.CDLL("libc.so.6")
        except Exception as e:
            self.libc = None
            logger.warning(f"⚠️ 无法加载 libc.so.6，malloc_trim 将被禁用: {e}")

    def _get_ram_usage_gb(self):
        """获取当前进程的实际物理内存占用 (RSS)"""
        if self.process:
            return self.process.memory_info().rss / (1024 ** 3)
        return 0.0

    def _do_thorough_cleanup(self, state, trigger_name):
        """执行彻底的同步清理流程"""
        
        # 判断是否为分布式环境
        is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        # 仅在主进程打印日志，避免 64 卡日志刷屏
        is_main_process = (not is_dist) or (torch.distributed.get_rank() == 0)

        # ==========================================
        # 1. 战前集合：同步所有进程，防止时序错位导致死锁
        # ==========================================
        if is_dist:
            torch.distributed.barrier()

        if is_main_process:
            ram_before = self._get_ram_usage_gb()
            vram_before = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0
            logger.info(f"====== 🧹 [{trigger_name}] 深度内存清理 (Step {state.global_step}) ======")
            start_time = time.time()

        # ==========================================
        # 2. Python 层：清理跨代循环引用对象
        # ==========================================
        collected_objs = 0
        for _ in range(3): 
            n = gc.collect(2) # 强制执行第 2 代全量回收
            collected_objs += n
            if n == 0:
                break # 扫干净了就提前退出

        # ==========================================
        # 3. PyTorch 层：清空底层的缓存池 (非常关键)
        # ==========================================
        if torch.cuda.is_available():
            torch.cuda.empty_cache()       # 清理未使用的显存碎片
            torch.cuda.ipc_collect()       # 清理多进程间通信 (IPC) 的残留内存映射
            
        # 尝试清理 PyTorch CPU 端的分配器缓存（部分版本支持）
        if hasattr(torch, '_C') and hasattr(torch._C, '_clear_allocator_cache'):
            torch._C._clear_allocator_cache()

        # ==========================================
        # 4. OS/C 层：强迫 glibc 将空闲内存归还给 Linux
        # ==========================================
        if self.libc is not None:
            self.libc.malloc_trim(0)

        # ==========================================
        # 5. 战后集合：等所有人打扫完，再一起继续训练
        # ==========================================
        if is_dist:
            torch.distributed.barrier()

        # ==========================================
        # 6. 统计与汇报
        # ==========================================
        if is_main_process:
            elapsed = time.time() - start_time
            ram_after = self._get_ram_usage_gb()
            vram_after = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0
            
            ram_freed = ram_before - ram_after
            vram_freed = vram_before - vram_after
            
            # 为了防止浮点误差导致显示负数，做个小处理
            ram_freed = max(0.0, ram_freed)
            vram_freed = max(0.0, vram_freed)
            
            logger.info(f"✅ 清理耗时: {elapsed:.2f}s | 销毁循环对象: {collected_objs} 个")
            logger.info(f"📉 系统 RAM: {ram_before:.2f} GB -> {ram_after:.2f} GB (释放: {ram_freed:.2f} GB)")
            logger.info(f"📉 GPU 显存: {vram_before:.2f} GB -> {vram_after:.2f} GB (释放: {vram_freed:.2f} GB)")
            logger.info("================================================================")

    def on_step_end(self, args, state, control, **kwargs):
        # 周期性计划清理
        if state.global_step > 0 and state.global_step % self.full_gc_interval == 0:
            self._do_thorough_cleanup(state, trigger_name="Scheduled")

    def on_save(self, args, state, control, **kwargs):
        # 保存 Checkpoint 时的伴生清理
        # 注意：Trainer 会在 save 前后调用此方法，正好把 save 产生的 CPU 内存突增削掉
        self._do_thorough_cleanup(state, trigger_name="Checkpoint")

class AdvancedMemoryTracker(TrainerCallback):
    """
    针对深度学习训练设计的专业内存/显存监控回调。
    支持：物理内存追踪、显存细分分析、内存增长计算 (Delta) 以及潜在泄漏预警。
    """
    def __init__(self, log_interval=100, growth_warn_threshold_mb=1000):
        self.log_interval = log_interval
        # 当单次间隔内存增长超过该阈值时，触发 Warning
        self.growth_warn_threshold_mb = growth_warn_threshold_mb 
        self.process = psutil.Process(os.getpid())
        
        # 历史状态记录，用于计算趋势 (Delta)
        self.last_cpu_ram_mb = 0
        self.last_gpu_alloc_mb = 0
        
        # 峰值记录
        self.peak_cpu_ram_mb = 0
        self.peak_gpu_alloc_mb = 0

    def _get_cpu_ram(self):
        # 获取当前进程占用的物理内存 RSS (Resident Set Size)
        return self.process.memory_info().rss / (1024 ** 2)

    def _get_gpu_stats(self):
        if not torch.cuda.is_available():
            return None
        
        device = torch.cuda.current_device()
        stats = torch.cuda.memory_stats(device)
        
        return {
            "alloc_mb": stats.get("active_bytes.all.current", 0) / (1024**2),
            "peak_alloc_mb": stats.get("active_bytes.all.peak", 0) / (1024**2),
            "rsvd_mb": stats.get("reserved_bytes.all.current", 0) / (1024**2),
            "peak_rsvd_mb": stats.get("reserved_bytes.all.peak", 0) / (1024**2),
            # inactive_split_bytes 极具参考价值，数值过大说明显存严重碎片化
            "inactive_split_mb": stats.get("inactive_split_bytes.all.current", 0) / (1024**2),
            # 显存分配重试次数，如果不为0并在增长，说明即将 OOM
            "retries": stats.get("num_alloc_retries", 0)
        }

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_interval != 0:
            return
            
        # 1. 采集并计算 CPU 物理内存
        cpu_ram_mb = self._get_cpu_ram()
        cpu_delta = cpu_ram_mb - self.last_cpu_ram_mb if self.last_cpu_ram_mb > 0 else 0
        self.peak_cpu_ram_mb = max(self.peak_cpu_ram_mb, cpu_ram_mb)
        
        log_str = f"\n=== [Step {state.global_step}] Memory Diagnostics ===\n"
        log_str += f"💻 [CPU RAM] Current: {cpu_ram_mb:.2f} MB | Delta: {cpu_delta:+.2f} MB | Peak: {self.peak_cpu_ram_mb:.2f} MB\n"
        
        # 2. 采集并计算 GPU 显存
        gpu_stats = self._get_gpu_stats()
        gpu_delta = 0
        if gpu_stats:
            gpu_alloc_mb = gpu_stats["alloc_mb"]
            gpu_delta = gpu_alloc_mb - self.last_gpu_alloc_mb if self.last_gpu_alloc_mb > 0 else 0
            self.peak_gpu_alloc_mb = max(self.peak_gpu_alloc_mb, gpu_alloc_mb)
            
            log_str += f"🎮 [GPU VRAM] Alloc: {gpu_alloc_mb:.2f} MB | Delta: {gpu_delta:+.2f} MB | Peak Alloc: {gpu_stats['peak_alloc_mb']:.2f} MB\n"
            log_str += f"             Rsvd: {gpu_stats['rsvd_mb']:.2f} MB | Peak Rsvd: {gpu_stats['peak_rsvd_mb']:.2f} MB\n"
            log_str += f"🔍 [GPU Health] Fragmentation (Inactive Split): {gpu_stats['inactive_split_mb']:.2f} MB | OOM Retries: {gpu_stats['retries']}"
            
        logger.info(log_str)

        # 3. 内存泄漏预警机制
        if self.last_cpu_ram_mb > 0 and (cpu_delta > self.growth_warn_threshold_mb or gpu_delta > self.growth_warn_threshold_mb):
            logger.warning(
                f"⚠️ [Memory Alert] 发现显著的内存/显存增长! "
                f"CPU 增长: {cpu_delta:+.2f} MB, GPU 增长: {gpu_delta:+.2f} MB。"
                f"请检查是否存在张量未 detech() 或 DataLoader 泄漏。"
            )
            
        # 4. 更新历史记录
        self.last_cpu_ram_mb = cpu_ram_mb
        if gpu_stats:
            self.last_gpu_alloc_mb = gpu_alloc_mb

def train():
    parser = HfArgumentParser(TrainConfig)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (training_args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (training_args,) = parser.parse_args_into_dataclasses()

    training_args.output_dir = broadcast_object_list([training_args.output_dir])[0]
    logger.info(f"set output-dir to {training_args.output_dir}")

    # configure processor
    processor = PRTS_Qwen3VLProcessor.from_pretrained(
        training_args.processor_name_or_path,
        padding_side="right",
    )
    if not training_args.train_mm_only:
        action_tokenizer = UniversalActionProcessor.from_pretrained(
            training_args.action_tokenizer_path, 
            trust_remote_code=True
        )
        processor.set_action_tokenizer(action_tokenizer)

    # load model config
    default_model_config_path = os.path.join(os.path.dirname(__file__), '..', 'prts_weights', 'config_qwen3_vl.json')
    config = PRTS_FlowMatchingConfig_Qwen3VL.from_pretrained(
        default_model_config_path if training_args.model_name_or_path is None \
            else training_args.model_name_or_path,
    )
    config.dit_action_head_config["use_mot_action_expert"] = training_args.use_mot_action_expert
    config.max_action_dim = training_args.max_action_dim
    config.action_chunk_size = training_args.chunk_size
    config.embodiment_tag = training_args.embodiment_tag
    logger.info(f"Set embodiment_tag='{training_args.embodiment_tag}' into config")

    if training_args.train_action_expert:  # post-training phase: enable DiT action head
        config.flow_matching_action_loss_weight = 1.0
        # config.dit_action_head_config["dit_model_type"] = training_args.dit_model_type

    ## set special token in the config
    for key, token in processor.token_ids.items():
        setattr(config, key, token)
        setattr(config.text_config, key, token)

    # configure model
    compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model_pretrained_path = training_args.vlm_name_or_path if training_args.model_name_or_path is None else training_args.model_name_or_path

    model, loading_info = PRTS_Qwen3VL.from_pretrained(
        model_pretrained_path,
        config=config,
        dtype=compute_dtype,
        attn_implementation=training_args.attn_implementation,
        output_loading_info=True,
    )

    # IMPORTANT: from_pretrained calls _init_weights on modules whose parameters
    # are absent from the checkpoint, overwriting custom initialization.
    # Re-apply proper init for all non-checkpoint modules when loading from base VLM.
    if config.flow_matching_action_loss_weight > 0. and training_args.model_name_or_path is None:
        logger.info("Trigger ``reset_parameters`` func of dit_action_head")
        model.dit_action_head.reset_parameters()

    # Freeze CRL heads during SFT: the released checkpoint already contains a
    # well-initialized CRL value head; we keep its weights for downstream value
    if hasattr(model, "crl_action_encoder") and hasattr(model, "crl_goal_encoder"):
        model.frozen_crl_modules()

    if not training_args.train_mm_only:
        model.set_fast_action_info(processor.action_mapper, processor.action_token_start_index)

    ## Resize embeddings if needed
    do_resize_embeddings = (len(processor.tokenizer) > model.config.vocab_size)
    if do_resize_embeddings:
        logger.info(f"Resizing model embeddings from {model.config.vocab_size} to {len(processor.tokenizer)}")
        model.resize_token_embeddings(len(processor.tokenizer))
        model.config.vocab_size = model.config.text_config.vocab_size

    # configure model
    configure_llm_qwen3_vl(model, training_args)
    configure_vision_tower(model, training_args, compute_dtype, training_args.device)

    # load dataset
    logger.info("Loading dataset...", main_process_only=True)
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    if training_args.use_fake_dataset and training_args.fake_dataset_mode == "reuse":
        data_module = {"train_dataset": None, "eval_dataset": None, "data_collator": None}
    else:
        time_start = time.time()
        data_module = make_dataset(processor=processor, args=training_args, rank=local_rank)
        logger.info(f"[Local Rank {local_rank}]: dataset loading time: {time.time() - time_start} seconds")
        # Wait for everyone
        if dist.is_initialized():
            dist.barrier()

    # DiT action head uses cross-attention (not KV cache), so disable cache for training
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # trainable params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.warning(
        f"{total_params=}, {trainable_params=}, [{trainable_params / total_params * 100}%]",
        main_process_only=True,
    )
    
    # aggregate data lengths for packing
    if data_module["train_dataset"] is not None and training_args.pack_dataset:
        dataset = data_module["train_dataset"]

        pack_meta = {}
        if local_rank == 0:
            time_start = time.time()
            if training_args.pack_indices_path and os.path.exists(training_args.pack_indices_path):
                logger.info(f"loading packed dataset from: {training_args.pack_indices_path}", main_process_only=True)
                pack_meta["mode"] = "ultra"
                pack_meta["grouped_indices"] = training_args.pack_indices_path
                pack_meta["lengths"] = os.path.join(os.path.dirname(training_args.pack_indices_path), 'data_lengths.npz')
            else:
                pack_meta = pack_dataset(
                    dataset=dataset,
                    args=training_args,
                    seed=42, 
                    memory_mode="ultra"
                )
                logger.info(f"packing time: {time.time() - time_start} seconds", main_process_only=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        time_start = time.time()
        pack_meta = broadcast_object_list([pack_meta])[0]
        packed_dataset = PackedDataset(
            dataset=dataset, 
            pack_length=training_args.max_packed_length, 
            mini_vqa_set_length=training_args.mini_vqa_set_length, 
            pack_meta=pack_meta
        )
        logger.info(f"[Local Rank {local_rank}]: packed dataset created in {time.time() - time_start} seconds")
        # Wait for everyone
        if dist.is_initialized():
            dist.barrier()

        # 更新data_module中的训练数据集
        data_module["train_dataset"] = packed_dataset
        dataset = packed_dataset

        # replace qwen2.5-vl attention class
        assert training_args.attn_implementation == "flash_attention_3" or training_args.attn_implementation == "flash_attention_2"
        assert training_args.train_action_expert == False, "DiT action head requires per-sample hidden states and is not compatible with packed dataset"
        replace_qwen2_vl_attention_class()
        
    elif data_module["train_dataset"] is not None:
        dataset = data_module["train_dataset"]
    else:
        dataset = None

    if training_args.use_fake_dataset and training_args.fake_dataset_mode == "measure" and dataset is not None:
        save_dir = training_args.fake_dataset_save_dir or os.path.join(training_args.output_dir, "reused_inputs_cache")
        os.makedirs(save_dir, exist_ok=True)
        rank_id = dist.get_rank() if dist.is_initialized() else 0
        file_path = os.path.join(save_dir, f"rank_{rank_id}_all_inputs.pt")

        def _to_cpu(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu()
            elif isinstance(x, dict):
                return {k: _to_cpu(v) for k, v in x.items()}
            elif isinstance(x, (list, tuple)):
                return type(x)(_to_cpu(v) for v in x)
            else:
                return x

        from torch.utils.data import DataLoader
        collate_fn = data_module["data_collator"]
        per_device_bs = training_args.per_device_train_batch_size
        num_workers = training_args.dataloader_num_workers
        dl = DataLoader(dataset, batch_size=per_device_bs, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

        target_steps = getattr(training_args, "fake_dataset_num_steps", None)
        if target_steps is not None and isinstance(target_steps, int) and target_steps > 0:
            measure_count = int(target_steps) * int(training_args.gradient_accumulation_steps)
        else:
            measure_count = int(training_args.fake_dataset_num_batches)

        inputs_buffer = []
        for i, batch in enumerate(dl):
            inputs_buffer.append(_to_cpu(batch))
            if i + 1 >= measure_count:
                break

        torch.save(inputs_buffer, file_path)
        if dist.is_initialized():
            dist.barrier()
        training_args.fake_dataset_mode = "reuse"
        data_module = {"train_dataset": None, "eval_dataset": None, "data_collator": None}

        del inputs_buffer
        del dl
        if 'batch' in locals():
            del batch

    model.config.use_cache = False
    model.language_model.config.use_cache = False

    trainer = PRTSTrainer(
        model=model,
        args=training_args,
        processing_class=processor,
        **data_module,
    )

    if trainer.accelerator.is_main_process and dataset is not None:
        dataset.info_qwen_vision_fetch()
        input_ids = dataset[0]["input_ids"]
        if input_ids.ndim == 2:
            logger.info(f"sample: \n{processor.tokenizer.decode(input_ids[0])}", main_process_only=True)
        else:
            logger.info(f"sample: \n{processor.tokenizer.decode(input_ids)}", main_process_only=True)

    trainer.add_callback(EnhancedGCCallback(full_gc_interval=2000))
    trainer.add_callback(MemoryMonitorCallback(log_interval=1))
    trainer.add_callback(AdvancedMemoryTracker(log_interval=100, growth_warn_threshold_mb=1000))

    del dataset
    if 'packed_dataset' in locals():
        del packed_dataset
        
    gc.collect()
    gc.freeze()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        logger.info("resume from checkpoint")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    model.config.use_cache = True
    trainer.save_state()
    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=f"{training_args.output_dir}/checkpoint-final-{trainer.state.global_step}"
    )


if __name__ == "__main__":
    train()
