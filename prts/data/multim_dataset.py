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

import copy
import math
import random
import re

import torch
import ujson as json
from lerobot.constants import ACTION, OBS_STATE
from torch.utils.data import Dataset
import torch.distributed as dist

from ..constants import (
    ACTION_END_TOKEN,
    ACTION_START_TOKEN,
    ACTION_PLACEHOLDER_TOKEN,
    IMAGE_PLACEHOLDER_TOKEN,
    VIDEO_PLACEHOLDER_TOKEN,
    LLAVA_ACTION_TOKEN,
    LLAVA_IMAGE_TOKEN,
    LLAVA_VIDEO_TOKEN,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
    ## training phase
    PRETRAINING_PHASE,
    POSTTRAINING_PHASE
)
from .lerobot_dataset import MultiLeRobotDataset
from .schema import MMDatasetConfig
from prts.utils.logging_utils import (
    normal,
    info,
    success,
    warning,
    error,
    fail,
    debug,
    important,
    cprint
)

local_rank = dist.get_rank() if dist.is_initialized() else 0

def rank0_print(message, msg_type='normal', *args):
    if local_rank == 0:
        cprint(message=message, msg_type=msg_type, *args)

class MultimodalDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_configs: list[MMDatasetConfig],
        max_num_tokens: int = 5120,
        meta_dataset: MultiLeRobotDataset = None,
        max_action_dim: int = 32,
        chunk_size: int = 50,
        sample_actions: bool = True,
        phase: str = PRETRAINING_PHASE,
        rank: int = 0,
    ):
        global local_rank
        local_rank = rank
        super().__init__()
        self.data_configs = data_configs
        self.max_action_dim = max_action_dim
        self.chunk_size = chunk_size
        self.enable_denoising_trigger = (phase == POSTTRAINING_PHASE)

        self.collection_names = [dt_config.name for dt_config in data_configs]

        list_data_dict, dataset_lens = [], []
        for i, dataset in enumerate(data_configs):
            json_path = dataset.json_path
            sampling_strategy = dataset.sampling_strategy
            sampling_number = None

            if json_path.endswith(".jsonl"):
                cur_data_dict = []
                with open(json_path, 'r') as f:  # ✅
                    for line in f:
                        cur_data_dict.append(json.loads(line.strip()))
            elif json_path.endswith(".json"):
                with open(json_path, 'r') as f:  # ✅
                    cur_data_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported file type: {json_path}")

            # NOTE: filter out lines above MAX_SEQ_LENGTH
            cur_data_dict = [line for line in cur_data_dict if line.get("num_tokens", 0) <= max_num_tokens]

            if ":" in sampling_strategy:
                sampling_strategy, sampling_number = sampling_strategy.split(":")
                if "%" in sampling_number:
                    sampling_number = math.ceil(
                        float(sampling_number.split("%")[0]) * len(cur_data_dict) / 100
                    )
                else:
                    sampling_number = int(sampling_number)

            # sampling
            if sampling_strategy == "first" and sampling_number is not None:
                cur_data_dict = cur_data_dict[:sampling_number]
            elif sampling_strategy == "end" and sampling_number is not None:
                cur_data_dict = cur_data_dict[-sampling_number:]
            elif sampling_strategy == "random" and sampling_number is not None:
                random.shuffle(cur_data_dict)
                cur_data_dict = cur_data_dict[:sampling_number]

            rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}", msg_type='success')
            dataset_lens.append(len(cur_data_dict))
            for data in cur_data_dict:
                data["vision_base_idx"] = i     # NOTE (zy): 标志来自第几个mm dataset的vision数据
                data["dataset_names"] = dataset.name
            list_data_dict.extend(cur_data_dict)

        self.data = list_data_dict
        self.dataset_lens = dataset_lens
        self.__set_metadata(meta_dataset)

        # set to false during calculating lengths
        self.sample_actions = sample_actions

    def __len__(self):
        return len(self.data)

    def fetch_seq_length(self, idx):
        return self.data[idx].get("seq_length", None)

    def fetch_num_tokens(self, idx):
        return self.data[idx].get("num_tokens", None)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        sources = self.data[i]
        transformed_source = copy.deepcopy(sources)
        transformed_source["conversations"] = llava_to_openai(
            transformed_source["conversations"], "video" in sources
        )
        return transformed_source

    def __get_metadata(self, repo_id, idx):
        """Get the metadata from lerobot dataset."""
        raise ValueError("__get_metadata has been deprecated")
        # return self.meta_dataset.getitem_by_id(repo_id, idx)
    
    def __get_metadata_ep_frame(self, repo_id, ep_idx: int, frame_idx: int):
        """Get the metadata from lerobot dataset."""
        return self.meta_dataset.getitem_by_ep_frame(repo_id, ep_idx, frame_idx)

    def __set_metadata(self, meta_dataset: MultiLeRobotDataset):
        """Set the metadata for the sources."""
        self.meta_dataset = meta_dataset

    @property
    def vision_base_paths(self):
        return [dataset.vision_base_path for dataset in self.data_configs]


def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r"\s*" + re.escape(LLAVA_VIDEO_TOKEN) + r"\n?"
        replacement = VISION_START_TOKEN + VIDEO_PLACEHOLDER_TOKEN + VISION_END_TOKEN
    else:
        pattern = r"\s*" + re.escape(LLAVA_IMAGE_TOKEN) + r"\n?"
        replacement = VISION_START_TOKEN + IMAGE_PLACEHOLDER_TOKEN + VISION_END_TOKEN
    return re.sub(pattern, replacement, input_string)


def replace_action_tokens(input_string):
    pattern = r"\s*" + re.escape(LLAVA_ACTION_TOKEN) + r"\n?"
    replacement = f"{ACTION_START_TOKEN}{ACTION_PLACEHOLDER_TOKEN}{ACTION_END_TOKEN}"
    return re.sub(pattern, replacement, input_string)


def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}
    transformed_data = []
    # 将conversations里面的内容全都替换掉
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_content = replace_action_tokens(transformed_content)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)
    return transformed_data


def pad_vector(vector, new_dim):
    """Can be (b s e) or (b e)"""
    if vector.shape[-1] == new_dim:
        return vector, torch.ones_like(vector, dtype=torch.bool).float()
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    vector_mask = torch.zeros_like(new_vector, dtype=torch.bool).float()

    new_vector[..., :current_dim] = vector
    vector_mask[..., :current_dim] = 1.0

    return new_vector, vector_mask
