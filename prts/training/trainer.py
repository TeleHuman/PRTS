# Copyright 2025 TeleAI Rhodes Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    is_sagemaker_mp_enabled,
    # logger,
)
from transformers.trainer_utils import (
    SaveStrategy,
    speed_metrics,
)
from ..data.config import ACTION_DATASET_NAMES, MULTIMODAL_DATASET_NAMES
from ..constants import POSTTRAINING_PHASE
from .train_utils import LOG_FORMATTER

import time
import os
import copy
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, Union, Optional
from accelerate.utils import DistributedType
import torch.profiler as profiler
from datetime import datetime

### for logging
import logging
from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")
if not logger.logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(LOG_FORMATTER)
    logger.logger.addHandler(handler)
###

PROFILE_START_STEP = 10
PROFILE_END_STEP = 14
LOG_DIR = "/path/to/prts_profiler_logs"
DO_PROFILE = False  # True

def identity_collate_fn(batch):
    return batch[0]

class MetaLossesTrainerState(TrainerCallback):
    """Callback to track meta losses (cross_entropy_loss and flow_loss) during training."""
    
    def __init__(self, meta_losses: list[str]):
        self.meta_losses = meta_losses

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.meta_losses = {k: torch.tensor(0.0).to(args.device) for k in self.meta_losses}
        control.channel_loss_dict = {
            "action_accuracy": torch.tensor(0.0, device=args.device)
        }
        control.channel_loss_count_dict = {
            "action_accuracy": torch.tensor(0, device=args.device)
        }

        # update fast tracking loss in ACTION_DATASET_NAMES
        control.channel_loss_dict.update(
            {
                f"fast/{dataset_name}": torch.tensor(0.0, device=args.device)
                for dataset_name in ACTION_DATASET_NAMES
            }
        )
        control.channel_loss_count_dict.update(
            {
                f"fast/{dataset_name}": torch.tensor(0, device=args.device)
                for dataset_name in ACTION_DATASET_NAMES
            }
        )

        if args.training_phase == POSTTRAINING_PHASE:
            # update flow matching tracking loss in ACTION_DATASET_NAMES
            control.channel_loss_dict.update(
                {
                    f"flow_matching/{dataset_name}": torch.tensor(0.0, device=args.device)
                    for dataset_name in ACTION_DATASET_NAMES
                }
            )
            control.channel_loss_count_dict.update(
                {
                    f"flow_matching/{dataset_name}": torch.tensor(0, device=args.device)
                    for dataset_name in ACTION_DATASET_NAMES
                }
            )
        
        # update ar tracking loss in MULTIMODAL_DATASET_NAMES
        control.channel_loss_dict.update(
            {
                f"vqa/{dataset_name}": torch.tensor(0.0, device=args.device)
                for dataset_name in MULTIMODAL_DATASET_NAMES
            }
        )
        control.channel_loss_count_dict.update(
            {
                f"vqa/{dataset_name}": torch.tensor(0, device=args.device)
                for dataset_name in MULTIMODAL_DATASET_NAMES
            }
        )

        ## V-- MODIFIED/ADDED --V ##
        # Add initialization for sampling time
        control.sampling_time_total = torch.tensor(0.0).to(args.device)
        # control.sampling_time_max = torch.tensor(-float('inf')).to(args.device)
        # control.sampling_time_min = torch.tensor(float('inf')).to(args.device)
        
        # Add initialization for forward time
        control.forward_time_total = torch.tensor(0.0).to(args.device)
        # control.forward_time_max = torch.tensor(-float('inf')).to(args.device)
        # control.forward_time_min = torch.tensor(float('inf')).to(args.device)

        # Add initialization for backward time
        control.backward_time_total = torch.tensor(0.0).to(args.device)
        # control.backward_time_max = torch.tensor(-float('inf')).to(args.device)
        # control.backward_time_min = torch.tensor(float('inf')).to(args.device)

        return control


class _TimingDataLoaderWrapper:
    """
    Wraps a DataLoader to time the __next__ call for batch sampling.
    Refactored as a generator to ensure multi-process iterator resources are 
    immediately garbage collected upon StopIteration.
    """
    def __init__(self, dataloader: DataLoader, trainer_control: TrainerControl):
        self.dataloader = dataloader
        self.trainer_control = trainer_control

    def __iter__(self):
        # 局部变量 iterator，当 yield 结束/跳出时，会自动触发 Python 局部变量回收
        iterator = iter(self.dataloader)
        while True:
            start_time = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            end_time = time.perf_counter()
            
            if hasattr(self.trainer_control, "sampling_time_total"):
                self.trainer_control.sampling_time_total.add_(end_time - start_time)
            
            yield batch
    
    def __len__(self):
        return len(self.dataloader)

    def __getattr__(self, name):
        return getattr(self.dataloader, name)

class ReusedInputsDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return copy.deepcopy(self.data[idx])

def worker_init_fn(worker_id):
    """
    DataLoader 子进程专属的初始化函数

    Note: This function no longer overrides the global garbage collection
    configuration, so worker processes follow the same GC behavior as the
    main process to avoid inconsistent memory management.
    """
    print(f"Worker {worker_id} started.")

class PRTSTrainer(Trainer):
    """Custom Trainer for PRTS model.
    
    This class extends the Trainer class from the transformers library to provide
    additional functionality specific to the PRTS model.
    
    It tracks and logs two main losses:
    - cross_entropy_loss: Standard language modeling loss
    - flow_loss: Flow matching loss for action prediction
    
    Args:
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        meta_losses = ["cross_entropy_loss", "flow_loss"]
        self.add_callback(MetaLossesTrainerState(meta_losses))

        # Profiler init
        if DO_PROFILE:
            # Ensure LOG_DIR is valid; fall back to a path under output_dir if not.
            _log_dir = LOG_DIR
            try:
                os.makedirs(_log_dir, exist_ok=True)
            except Exception:
                _log_dir = os.path.join(self.args.output_dir, "prts_profiler_logs")
                os.makedirs(_log_dir, exist_ok=True)

            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            self.prof_save_path = os.path.join(_log_dir, f"timeline_{dist.get_rank() if dist.is_initialized() else 0}_{timestamp}.json")

            def trace_handler(p):
                p.export_chrome_trace(self.prof_save_path)
            self.profiler = profiler.profile(
                activities=[
                    profiler.ProfilerActivity.CPU,
                    profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=True,
                on_trace_ready=trace_handler
            )
        else:
            self.profiler = None
        self._current_step = 0
        self._profiler_started = False

        # fake-dataset: strictly mimic fake_dataset_trainer.py
        self.reused_dataloader = None
        self.use_fake_dataset = getattr(self.args, "use_fake_dataset", False)
        self.fake_dataset_mode = getattr(self.args, "fake_dataset_mode", "reuse")
        self.fake_dataset_save_dir = getattr(self.args, "fake_dataset_save_dir", None) or os.path.join(self.args.output_dir, "reused_inputs_cache")
        self.fake_dataset_num_batches = getattr(self.args, "fake_dataset_num_batches", 8)

        if self.use_fake_dataset and self.fake_dataset_mode == "reuse":
            self._setup_reused_dataloader()

    # ── Norm-stats persistence ────────────────────────────────────────────────
    def _save_norm_stats(self, output_dir: str) -> None:
        """Save dataset norm stats to output_dir/norm_stats.json.

        Collects stats from every LeRobotDataset sub-dataset and writes them
        so PRTSInterface can reconstruct normalizer/unnormalizer without
        needing the original dataset path.
        """
        from ..data.normalize import save_norm_stats_for_model

        dataset = self.train_dataset
        lerobot_dataset = getattr(dataset, "lerobot_dataset", None)
        if lerobot_dataset is None:
            logger.warning("PRTSTrainer._save_norm_stats: train_dataset has no "
                           ".lerobot_dataset – skipping norm_stats.json")
            return

        sub_datasets = getattr(lerobot_dataset, "_datasets", None)
        if not sub_datasets:
            logger.warning("PRTSTrainer._save_norm_stats: lerobot_dataset._datasets "
                           "is empty – skipping norm_stats.json")
            return

        state_mode = self.args.state_mode
        state_mode_str = state_mode.value if hasattr(state_mode, "value") else str(state_mode)

        datasets_stats: dict = {}
        for ds in sub_datasets:
            try:
                raw_features = {
                    k: ds.meta.features[k]
                    for k in (ds.select_state_keys + ds.select_action_keys)
                    if k in ds.meta.features
                }
                raw_stats = {
                    k: dict(ds.meta.stats[k])
                    for k in (ds.select_state_keys + ds.select_action_keys)
                    if k in ds.meta.stats
                }
                datasets_stats[ds.repo_id] = {"features": raw_features, "stats": raw_stats}
            except Exception as e:
                logger.warning(f"PRTSTrainer._save_norm_stats: failed for {ds.repo_id}: {e}")

        if not datasets_stats:
            return

        try:
            save_norm_stats_for_model(output_dir, datasets_stats, state_mode=state_mode_str)
            logger.info(f"Saved norm_stats.json → {output_dir} ({len(datasets_stats)} dataset(s))")
        except Exception as e:
            logger.warning(f"PRTSTrainer._save_norm_stats: write failed: {e}")

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Override to co-save norm_stats.json alongside model weights."""
        super().save_model(output_dir, _internal_call)
        # Only the main process writes the norm stats file
        if self.args.local_rank not in (-1, 0):
            return
        
        if self.args.train_action_expert:
            self._save_norm_stats(output_dir or self.args.output_dir)
    
    def _save_launch_script(self):
        """
        Save the launch script to output directory for reproducibility.
        """
        import os
        import sys
        import shutil
        
        # Only save on main process
        if self.args.local_rank not in [-1, 0]:
            return
        
        # Get the command that launched the script
        # sys.argv[0] is the train.py script, but we want to find the .sh script
        # Check if there's a LAUNCH_SCRIPT environment variable
        launch_script_path = os.environ.get('LAUNCH_SCRIPT', None)
        
        if launch_script_path and os.path.exists(launch_script_path):
            try:
                script_name = os.path.basename(launch_script_path)
                dest_path = os.path.join(self.args.output_dir, script_name)
                shutil.copy2(launch_script_path, dest_path)
                logger.info(f"✅ Launch script saved to: {dest_path}")
            except Exception as e:
                logger.warning(f"Failed to copy launch script: {e}")
        else:
            # Try to infer from parent process or environment
            # Check common patterns for script detection
            try:
                import psutil
                current_process = psutil.Process(os.getpid())
                parent_process = current_process.parent()
                
                if parent_process:
                    cmdline = parent_process.cmdline()
                    # Look for .sh files in the command line
                    for arg in cmdline:
                        if arg.endswith('.sh') and os.path.exists(arg):
                            script_name = os.path.basename(arg)
                            dest_path = os.path.join(self.args.output_dir, script_name)
                            shutil.copy2(arg, dest_path)
                            logger.info(f"✅ Launch script saved to: {dest_path}")
                            return
            except ImportError:
                logger.debug("psutil not available, cannot auto-detect launch script")
            except Exception as e:
                logger.debug(f"Could not detect launch script: {e}")
            
            logger.info("💡 To save launch script, set LAUNCH_SCRIPT environment variable in your .sh file:\n"
                       "   export LAUNCH_SCRIPT=$(realpath $0)")

    def _save_optimizer_info(
        self, 
        optimizer, 
        opt_model, 
        visual_parameters, 
        merger_parameters, 
        action_parameters
    ):
        """
        Save optimizer configuration information to a file.
        
        Args:
            optimizer: The optimizer instance
            opt_model: The model being optimized
            visual_parameters: List of visual parameter names
            merger_parameters: List of merger parameter names
            action_parameters: List of action parameter names
        """
        import os
        
        # Only save on main process
        if self.args.local_rank not in [-1, 0]:
            return
            
        output_file = os.path.join(self.args.output_dir, "optim_info.txt")
        
        # Check if verbose mode is enabled
        verbose_optimizer_debug = os.environ.get('VERBOSE_OPTIMIZER_DEBUG', '0') == '1'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Helper function to write to file
            def write_line(line=""):
                f.write(line + "\n")
            
            write_line("=" * 80)
            write_line("🔍 Optimizer Configuration Summary")
            write_line("=" * 80)
            
            # Create a mapping from parameter id to name for debugging
            param_to_name = {id(p): n for n, p in opt_model.named_parameters()}
            
            total_params = 0
            for group_idx, group in enumerate(optimizer.param_groups):
                num_params = sum(p.numel() for p in group['params'])
                total_params += num_params
                
                lr = group.get('lr', self.args.learning_rate)
                wd = group.get('weight_decay', 0.0)
                group_name = group.get('name', f'group_{group_idx}')
                
                write_line()
                write_line(f"📦 Parameter Group {group_idx}: {group_name}")
                write_line(f"   Learning Rate: {lr:.2e}")
                write_line(f"   Weight Decay: {wd}")
                write_line(f"   Num Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
                
                # Show parameter names
                param_names = [param_to_name.get(id(p), 'unknown') for p in group['params']]
                if param_names:
                    if verbose_optimizer_debug:
                        # Print all parameter names
                        write_line(f"   All Parameters ({len(param_names)}):")
                        for pname in param_names:
                            write_line(f"      - {pname}")
                    else:
                        # Print only first 10 parameter names
                        write_line(f"   Sample Parameters:")
                        for pname in param_names[:10]:
                            write_line(f"      - {pname}")
                        if len(param_names) > 10:
                            write_line(f"      ... and {len(param_names) - 10} more")
                            write_line(f"      (Set VERBOSE_OPTIMIZER_DEBUG=1 to see all parameters)")
            
            write_line()
            write_line("=" * 80)
            write_line(f"📊 Total Trainable Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
            write_line(f"   Base LR: {self.args.learning_rate:.2e}")
            if self.args.vision_lr is not None:
                write_line(f"   Vision LR: {self.args.vision_lr:.2e} | Matched {len(visual_parameters)} params")
            if self.args.merger_lr is not None:
                write_line(f"   Merger LR: {self.args.merger_lr:.2e} | Matched {len(merger_parameters)} params")
            if self.args.action_lr is not None:
                write_line(f"   Action LR: {self.args.action_lr:.2e} | Matched {len(action_parameters)} params")
            write_line("=" * 80)
        
        logger.info(f"✅ Optimizer configuration saved to: {output_file}")

    def _fd_save_path(self):
        os.makedirs(self.fake_dataset_save_dir, exist_ok=True)
        rank_id = dist.get_rank() if dist.is_initialized() else 0
        return os.path.join(self.fake_dataset_save_dir, f"rank_{rank_id}_all_inputs.pt")

    def _setup_reused_dataloader(self):
        file_path = self._fd_save_path()
        if not os.path.exists(file_path):
            # Warn but don't fail, maybe we are in measure mode or it will be created? 
            # But here we are in REUSE mode.
            logger.warning(f"fake-dataset missing: {file_path}. Cannot reuse.")
            return

        try:
            inputs_list = torch.load(file_path, map_location='cpu')
        except Exception as e:
            logger.error(f"Failed to load fake dataset: {e}")
            return

        ds = ReusedInputsDataset(inputs_list)
        self.reused_dataloader = DataLoader(
            ds,
            batch_size=1,
            shuffle=True, # Shuffle for training
            num_workers=0,
            collate_fn=identity_collate_fn,
        )
        logger.info(f"Reused dataloader setup complete. {len(inputs_list)} batches loaded.")

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        inputs = super()._prepare_inputs(inputs)
        return inputs
    

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)

            visual_parameters, merger_parameters, action_parameters = [], [], []
            if self.args.vision_lr is not None:
                visual_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "visual" in name and "merger" not in name
                ]
            if self.args.merger_lr is not None:
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]
            if self.args.action_lr is not None:
                action_parameters = [
                    name for name, _ in opt_model.named_parameters() if "dit_action_head" in name
                ]

            special_lr_parameters = merger_parameters + visual_parameters + action_parameters

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            if visual_parameters:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.vision_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.vision_lr,
                        },
                    ]
                )

            if merger_parameters:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.merger_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.merger_lr,
                        },
                    ]
                )

            if action_parameters:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (n in decay_parameters and n in action_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.action_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (n not in decay_parameters and n in action_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.action_lr,
                        },
                    ]
                )

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            # Save launch script for reproducibility
            self._save_launch_script()
            
            # Save optimizer configuration to file
            self._save_optimizer_info(
                self.optimizer, 
                opt_model, 
                visual_parameters, 
                merger_parameters, 
                action_parameters
            )

            if "bitsandbytes" in str(optimizer_cls) and optimizer_kwargs.get("optim_bits", None) == 8:
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss for PRTS model.
        
        The PRTS model returns PRTSModelOutputWithPast which contains:
        - loss: Combined total loss
        - cross_entropy_loss: Language modeling loss
        - flow_loss: Flow matching loss for actions
        
        This method accumulates the meta losses for logging.
        
        Args:
            model: PRTS model
            inputs: Input batch
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (unused)
            
        Returns:
            Loss tensor and optionally model outputs
        """
        if hasattr(self.control, "meta_losses") and model.training:
            
            # card_id = self.args.local_rank
            # global_rank = self.args.process_index
            # action_len = 0 if inputs.get('actions') is None else len(inputs.get('actions'))

            # print(
            #     f"[GPU-{card_id} | GlobalRank-{global_rank}] "
            #     f"len of actions: {action_len}"
            # )

            ## V-- MODIFIED/ADDED --V ##
            # 计时 Forward 过程 (即 super().compute_loss)
            start_time = time.perf_counter()
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            end_time = time.perf_counter()
            
            if hasattr(self.control, "forward_time_total"):
                self.control.forward_time_total.add_(end_time - start_time)

            ## A-- MODIFIED/ADDED --A ##

            if not isinstance(outputs, dict):
                raise ValueError(
                    "The model output should be a dictionary or ModelOutput and not a tuple or list."
                )

            # Accumulate meta losses for logging
            for k, v in outputs.items():
                if k in self.control.meta_losses and v is not None:
                    self.control.meta_losses[k].add_(v.detach().mean() / self.args.gradient_accumulation_steps)

            if outputs.get('channel_loss_dict', None) is not None:
                for k, v in outputs['channel_loss_dict'].items():
                    self.control.channel_loss_dict[k].add_(v.detach())

                for k, v in outputs['channel_loss_count_dict'].items():
                    self.control.channel_loss_count_dict[k].add_(v.detach())

            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)
        
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Overrides the base method to return a DataLoader wrapper that
        times batch sampling.
        """
        if self.use_fake_dataset and self.fake_dataset_mode == "reuse" and self.reused_dataloader is not None:
            return self.reused_dataloader
        original_dataloader = super().get_train_dataloader()
        original_dataloader.worker_init_fn = worker_init_fn
        
        # Wrap the dataloader and pass self.control for stat accumulation
        return _TimingDataLoaderWrapper(original_dataloader, self.control)

    ## used to implement our intentional logging
    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        """Override logging to include meta losses.
        
        This method extends the default logging to include cross_entropy_loss and flow_loss.
        """
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            # Log meta losses (cross_entropy_loss, flow_loss, mc_loss)
            if hasattr(self.control, "meta_losses"):
                for k, v in self.control.meta_losses.items():
                    logs[k] = self._nested_gather(v).mean().item()
                    self.control.meta_losses[k].zero_()
                    logs[k] = round(logs[k] / (self.state.global_step - self._globalstep_last_logged), 4)

            ### TODO: Review and improve channel loss calculation. The current implementation may not correctly average losses across all batches or processes. Verify that loss aggregation and division by count are accurate, and update the logic if necessary.
            # Log channel loss dict
            if hasattr(self.control, "channel_loss_dict"):
                for key in list(self.control.channel_loss_dict.keys()):
                    count_tensor = self.control.channel_loss_count_dict[key]
                    count_sum = self._nested_gather(count_tensor).sum().item()

                    if count_sum > 0:
                        loss_tensor = self.control.channel_loss_dict[key]
                        
                        channel_loss = self._nested_gather(loss_tensor).sum().item() / count_sum

                        if key == "action_accuracy":
                            logs[key] = round(channel_loss, 4)
                        else:
                            logs[f"{key}_channel_loss"] = round(channel_loss, 4)

                    self.control.channel_loss_dict[key].zero_()
                    self.control.channel_loss_count_dict[key].zero_()

            ## V-- MODIFIED/ADDED --V ##
            # Log sampling time
            if hasattr(self.control, "sampling_time_total"):
                sampling_time_tensor = self.control.sampling_time_total
                sampling_time = self._nested_gather(sampling_time_tensor) / (self.state.global_step - self._globalstep_last_logged)
                
                logs["time/avg_sampling_time_s"] = round(sampling_time.mean().item(), 4)
                logs["time/max_sampling_time_s"] = round(sampling_time.max().item(), 4)
                logs["time/min_sampling_time_s"] = round(sampling_time.min().item(), 4)

                # Reset the counters (in-place)
                self.control.sampling_time_total -= self.control.sampling_time_total

            # Log forward time
            if hasattr(self.control, "forward_time_total"):
                forward_time_tensor = self.control.forward_time_total
                forward_time = self._nested_gather(forward_time_tensor) / (self.state.global_step - self._globalstep_last_logged)
                
                logs["time/avg_forward_time_s"] = round(forward_time.mean().item(), 4)
                logs["time/max_forward_time_s"] = round(forward_time.max().item(), 4)
                logs["time/min_forward_time_s"] = round(forward_time.min().item(), 4)

                # Reset the counters
                self.control.forward_time_total -= self.control.forward_time_total

            # Log backward time
            if hasattr(self.control, "backward_time_total"):
                backward_time_tensor = self.control.backward_time_total
                backward_time = self._nested_gather(backward_time_tensor) / (self.state.global_step - self._globalstep_last_logged)
                
                logs["time/avg_backward_time_s"] = round(backward_time.mean().item(), 4)
                logs["time/max_backward_time_s"] = round(backward_time.max().item(), 4)
                logs["time/min_backward_time_s"] = round(backward_time.min().item(), 4)

                # Reset the counters
                self.control.backward_time_total -= self.control.backward_time_total

            ## A-- MODIFIED/ADDED --A ##

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def log(self, logs: dict[str, float], start_time = None) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                logs.update(speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen))

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _log_time_metrics(self, logs: dict[str, float], metric_name: str, curr_state_steps, last_state_steps):
        total_key = f"{metric_name}_time_total"
        max_key = f"{metric_name}_time_max"
        min_key = f"{metric_name}_time_min"

        delta_steps = curr_state_steps - last_state_steps
        
        if not hasattr(self.control, total_key):
            return

        total_time_tensor = getattr(self.control, total_key)
        total_time_sum = self._nested_gather(total_time_tensor).sum().item()
        avg_time_s = total_time_sum / delta_steps
        logs[f"time/avg_{metric_name}_s"] = round(avg_time_s, 4)

        # 3. 计算全局 Max
        max_time_tensor = getattr(self.control, max_key)
        global_max_s = self._nested_gather(max_time_tensor).max().item()
        logs[f"time/max_{metric_name}_s"] = round(global_max_s, 4)

        # 4. 计算全局 Min
        min_time_tensor = getattr(self.control, min_key)
        global_min_s = self._nested_gather(min_time_tensor).min().item()
        logs[f"time/min_{metric_name}_s"] = round(global_min_s, 4)

        # 5. 重置
        getattr(self.control, total_key).zero_()
        getattr(self.control, max_key).fill_(-float('inf'))
        getattr(self.control, min_key).fill_(float('inf'))

    ### override
    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Prepare buffers for context parallelism
        current_step = self._current_step
        if current_step == PROFILE_START_STEP and DO_PROFILE:
            print(f"*** Profiler STARTING at batch {current_step} ***")
            if self.profiler and not self._profiler_started:
                try:
                    self.profiler.start()
                    self._profiler_started = True
                except Exception as e:
                    print(f"[Profiler] start failed: {e}")
        
        if current_step == PROFILE_END_STEP and DO_PROFILE:
            print(f"*** Profiler STOPPING after batch {current_step} ***")
            if self.profiler and self._profiler_started:
                try:
                    self.profiler.stop()
                    self._profiler_started = False
                except Exception as e:
                    print(f"[Profiler] stop failed: {e}")
        self._current_step += 1
        
        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        # Context manager is no-op if CP isn't enabled
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)

            # --- Fake Dataset Collection (Measure Mode) ---
            if self.use_fake_dataset and self.fake_dataset_mode == "measure":
                if not hasattr(self, "measured_inputs"):
                    self.measured_inputs = []
                
                # Check if we reached the target number of batches
                if len(self.measured_inputs) < self.fake_dataset_num_batches:
                    # Collect current inputs (move to CPU to save memory and for serialization)
                    # Note: inputs are already processed by _prepare_inputs
                    inputs_cpu = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    
                    # Optional: Log shape for debugging (only first batch or specific steps)
                    if len(self.measured_inputs) == 0:
                         if "input_ids" in inputs_cpu:
                            logger.info(f"[Measure] input_ids shape: {inputs_cpu['input_ids'].shape}")
                         if "attention_mask" in inputs_cpu:
                            logger.info(f"[Measure] attention_mask shape: {inputs_cpu['attention_mask'].shape}")

                    self.measured_inputs.append(inputs_cpu)
                    logger.info(f"Collected fake dataset batch {len(self.measured_inputs)} / {self.fake_dataset_num_batches}")
                    
                    if len(self.measured_inputs) == self.fake_dataset_num_batches:
                        save_path = self._fd_save_path()
                        logger.info(f"Saving fake dataset to {save_path}...")
                        torch.save(self.measured_inputs, save_path)
                        logger.info("Fake dataset saved.")
                        # We could stop training here, but usually we just let it run or the user kills it
            # ---------------------------------------------

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                # Here we only consider using cuda
                torch.cuda.empty_cache()

            kwargs = {}

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            start_time = time.perf_counter()

            if self.use_apex:
                from apex import amp

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
                if (
                    not self.model_accepts_loss_kwargs or num_items_in_batch is None
                ) and self.compute_loss_func is None:
                    # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient accumulation steps
                    loss = loss / self.current_gradient_accumulation_steps

                # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
                # https://github.com/huggingface/transformers/pull/35808
                if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                    kwargs["scale_wrt_gas"] = False

                self.accelerator.backward(loss, **kwargs)

            end_time = time.perf_counter()
            
            if hasattr(self.control, "backward_time_total"):
                # 将时间累加到 control 对象上 (由 callback 初始化)
                self.control.backward_time_total.add_(end_time - start_time)

            return loss.detach()
