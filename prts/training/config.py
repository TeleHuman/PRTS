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

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# from lerobot.configs.types import NormalizationMode
from ..data.normalize import NormalizationMode
from transformers import TrainingArguments
from ..constants import PRETRAINING_PHASE, POSTTRAINING_PHASE


@dataclass
class TrainConfig(TrainingArguments):
    """qwen2.5-vl vision parameters"""

    image_min_pixels: int | None = field(default=64 * 28 * 28)      # min number of tokens = 64
    image_max_pixels: int | None = field(default=128 * 28 * 28)     # max number of tokens = 256
    video_min_pixels: int | None = field(default=64 * 28 * 28)      # min number of tokens = 64
    video_max_pixels: int | None = field(default=128 * 28 * 28)     # max number of tokens = 256
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    video_resized_width: int = field(default=None)
    video_resized_height: int = field(default=None)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    fps: float = 1.0

    """dataset parameters"""
    data_path: str = field(default=None, metadata={"help": "Path to training data or yaml config."})
    rope_type: str = field(default="qwen2_5_vl", metadata={"help": "Type of rope."})
    train_mm_only: bool = False
    train_lerobot_only: bool = False
    training_phase: str = PRETRAINING_PHASE     # or POSTTRAINING_PHASE

    lerobot_data_video_backend: str | None = "torchcodec"
    state_mode: NormalizationMode | None = NormalizationMode.MEAN_STD
    pack_dataset: bool = False
    pack_indices_path: str | None = field(default=None)
    max_packed_length: int = field(default=768, metadata={"help": "Maximum sequence length."})
    mini_vqa_set_length: int = field(
        default=256, metadata={"help": "Maximum length of mini vqa set data in dataset packing."}
    )

    """ model parameters """
    model_name_or_path: str | None = field(default=None)
    vlm_name_or_path: str | None = field(default=f"{os.environ['HF_HUB_CACHE']}/models--Qwen--Qwen2.5-VL-3B-Instruct")
    processor_name_or_path: str | None = field(default=None)
    action_tokenizer_path: str | None = field(default=f"{os.environ['HF_HUB_CACHE']}/models--physical-intelligence--fast")
    
    chunk_size: int | None = 16
    max_action_dim: int | None = 48

    train_action_expert: bool | None = None                 # This argument is automatically initialized in __post_init__. Do not manually set it.
    ratio_fast_action_tokens: float = field(default=0.)     # only valid when train_action_expert is True
    # dit_model_type: str = field(default="DiT-B", metadata={"help": "DiT action head preset: DiT-B (768d, 12h) or DiT-L (1536d, 32h)"})

    """ training parameters """
    cache_dir: str | None = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    freeze_lm_head: bool = field(default=False)
    attn_implementation: str = field(default="sdpa")  # sdpa, flash_attention_2, flash_attention_3

    # lora_enable: bool = False
    # vision_lora: bool = False
    # use_dora: bool = False
    # lora_rank: int = 64
    # lora_alpha: int = 16
    # lora_dropout: float = 0.05
    # lora_weight_path: str = ""
    # lora_bias: str = "none"

    vision_lr: float | None = None
    merger_lr: float | None = None
    action_lr: float | None = None
    lora_namespan_exclude: str = field(
        default=None, metadata={"help": "List of namespan to exclude for LoRA"}
    )
    num_lora_modules: int = -1

    gamma: float = field(
        default=0.99,
        metadata={"help": "Discount factor"},
    )

    """experiment parameters"""
    output_base: str = field(default="outputs", metadata={"help": "Base directory for output."})

    # Embodiment tag for finetuning. When set, it is written into the saved model
    # config (model.config.embodiment_tag) so that eval code can read the
    # delta_action_mask without needing the training dataset config.
    embodiment_tag: str | None = field(
        default=None,
        metadata={"help": "Embodiment tag identifying robot type (see prts/data/embodiment_tag.py). Should be same as the data config you are using during post-training."},
    )
    use_mot_action_expert: bool = field(
        default=False,
        metadata={"help": "Whether to use MoT action expert"},
    )

    """fake dataset parameters"""
    use_fake_dataset: bool = field(default=False, metadata={"help": "Use pre-recorded inputs instead of loading dataset"})
    fake_dataset_mode: str = field(default="reuse", metadata={"help": "measure|reuse. 'measure' records a few real batches then reuses; 'reuse' loads existing saved batches"})
    fake_dataset_save_dir: Optional[str] = field(default=None, metadata={"help": "Directory to save/load reused inputs, defaults to <output_dir>/reused_inputs_cache"})
    fake_dataset_num_batches: int = field(default=8, metadata={"help": "Number of batches to record when mode=measure"})
    fake_dataset_num_steps: Optional[int] = field(default=None, metadata={"help": "Target optimizer steps to record in measure mode"})

    def __post_init__(self):
        super().__post_init__()

        """check validity"""
        if self.train_lerobot_only and self.train_mm_only:
            self.train_mm_only = False
            warnings.warn("`train_mm_only` is set to False when `train_lerobot_only` is True.", stacklevel=2)

        # if self.lora_enable and not self.freeze_llm:
        #     self.freeze_llm = True
        #     warnings.warn("`freeze_llm` is set to True when `lora_enable`.", stacklevel=2)

        # if not self.lora_enable and self.vision_lora:
        #     self.vision_lora = False
        #     warnings.warn("`vision_lora` is set to False when `lora_enable` is False.", stacklevel=2)

        # if self.vision_lora and not self.freeze_vision_tower:
        #     self.freeze_vision_tower = True
        #     warnings.warn("`freeze_vision_tower` is set to True when `vision_lora` is True.", stacklevel=2)

        if self.training_phase == PRETRAINING_PHASE:
            self.train_action_expert = False
        else:
            self.train_action_expert = True

        if self.processor_name_or_path is None:
            self.processor_name_or_path = self.model_name_or_path or self.vlm_name_or_path

        if self.output_dir == "trainer_output":
            import datetime as dt

            self.output_dir = f"{self.output_base}/{dt.datetime.now():%Y-%m-%d/%H-%M-%S}-{self.run_name}"
