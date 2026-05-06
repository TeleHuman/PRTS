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

"""This module provides custom LeRobot dataset implementations that extend the base classes from the `lerobot` library.

The `LeRobotDataset` class adds features such as:
- Subtask training modes.
- Selection of specific video, state, and action keys.
- Dataset weighting for sampling.
- Delta action calculation.
- State and action normalization.

The `MultiLeRobotDataset` class is a wrapper for loading and combining multiple `LeRobotDataset` instances,
potentially from different repositories. It supports parallel data loading to speed up the process.
"""

import time
import bisect
import multiprocessing
import os
import random
import numpy as np
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import datasets
import torch
from datasets import load_dataset
from lerobot.constants import ACTION, HF_LEROBOT_HOME, OBS_STATE
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset as BaseLeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset as BaseMultiLeRobotDataset,
)
from lerobot.datasets.utils import hf_transform_to_torch, serialize_dict
from torch.utils.data import ConcatDataset
from tqdm import tqdm
import torch.distributed as dist

import json

from .schema import LerobotConfig
from .embodiment_tag import EMBODIMENT_CONFIGS
from .transforms import dataset_to_policy_features, DeltaActionTransform
from .normalize import load_norm_stats, NormalizationMode, Normalize, Unnormalize
from prts.utils.logging_utils import cprint
"""lerobot datasets"""

local_rank = dist.get_rank() if dist.is_initialized() else 0

def rank0_print(message, msg_type='normal', *args):
    if local_rank == 0:
        cprint(message=message, msg_type=msg_type, *args)

CODE_VERSION = "v2.1"

class LeRobotDataset(BaseLeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        # custom features
        state_mode: str = "MEAN_STD",
        select_video_keys: list[str] | None = None,
        select_state_keys: list[str] | None = None,
        select_action_keys: list[str] | None = None,
        train_subtask: str | None = None,  # ["cumulate", "mixture:0.5", "true"]
        load_quantile_stats: bool = False,
        # state-relative action (GR00T-style): action[t] - state[t] for masked dims
        state_relative_action: bool = False,
        # embodiment tag: identifies robot type and its delta_action_mask
        embodiment_tag: str | None = None,
        # weight: float | None = None,
        rank: int = 0
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
        )
        self.rank = rank
        # store embodiment tag for delta_action_mask lookup and model config propagation
        self.embodiment_tag = embodiment_tag

        # remove unused features for efficiency
        self.set_feature_keys(select_video_keys, select_state_keys, select_action_keys)

        # calculate substake indices
        self.set_train_subtask(train_subtask)

        # set episode from index
        self.get_episode_from_index(episodes)

        # load dataset norm stats from pre-computed json file
        self.load_quantile_stats = load_quantile_stats
        if self.load_quantile_stats:
            self.load_dataset_norm_stats()

        # state-relative action: override action stats with state-relative stats,
        # then apply transform in post_process before normalization.
        # Must be called AFTER load_dataset_norm_stats and BEFORE set_normalization.
        self.set_state_relative_action(state_relative_action)

        # set nomalizer for multiple datasets (should be called after load quantile stats)
        self.set_normalization(state_mode)

    def load_dataset_norm_stats(self):
        norm_stats_path = self.root / 'meta' / f'norm_stats.json'

        if not norm_stats_path.exists():
            raise ValueError(f"\033[1;33m ⚠️ No norm stats files found in the path \'{norm_stats_path}\'. \n Please analyze lerobot dataset first to get norm stats! \033[0m")
        
        else:
            norm_stats = load_norm_stats(self.root)

            q_existence = norm_stats[self.select_action_keys[0]]['q01'] is not None and \
                norm_stats[self.select_action_keys[0]]['q99'] is not None
            if not q_existence:
                raise ValueError("\033[1;33m ⚠️ Quantile Normalization factors missing! \033[0m")
            
            for key, value in norm_stats.items():
                for item, stats in value.items():
                    self.meta.stats[key][item] = stats # directly update the stats

    def set_train_subtask(self, train_subtask: str | None = None):
            """set train subtask mode for lerobot dataset"""
            self.train_subtask = train_subtask
            if train_subtask is None:
                return

            task_sizes = {}
            has_valid_config = False

            try:
                for _, ep in self.meta.episodes.items():
                    # 获取 action_config，如果不存在则默认为 None
                    config = ep.get("action_config")
                    ep_idx = ep["episode_index"]

                    if config is None:
                        task_sizes[ep_idx] = []
                        continue

                    # 策略 1: 标准列表结构 (直接获取 end_frame)
                    if isinstance(config, list):
                        task_sizes[ep_idx] = [item["end_frame"] for item in config]
                        has_valid_config = True

                    # 策略 2: 字典结构 (需要从字符串解析数字)
                    elif isinstance(config, dict) and "steps" in config:
                        # 修复了原代码中 filter 不能直接切片的问题
                        # 逻辑：提取字符串中的所有数字并转为 int
                        parsed_steps = []
                        for item in config["steps"]:
                            num_str = ''.join(filter(str.isdigit, str(item["end_frame"])))
                            if num_str:
                                parsed_steps.append(int(num_str))
                        task_sizes[ep_idx] = parsed_steps
                        has_valid_config = True
                    
                    # 如果既不是 list 也不是带 steps 的 dict，视为空
                    else:
                        task_sizes[ep_idx] = []

                # 如果遍历完发现没有任何有效的 config，则关闭 subtask
                if not has_valid_config:
                    self.train_subtask = None

            except Exception as e:
                cprint(f"[Local Rank {self.rank}] {self.repo_id} failed to calculate episode subtask cumulate: {e}", msg_type='warning')
                self.train_subtask = None
                task_sizes = {} # 出错时清空，避免数据不一致

            self.task_sizes = task_sizes
            cprint(f"[Local Rank {self.rank}]: * set train_subtask {self.train_subtask} for {self.repo_id}", msg_type='info')

    def set_feature_keys(self, video_keys=None, state_keys=None, action_keys=None):
        """select video, state and action keys from the dataset"""
        self.select_video_keys = video_keys or self.meta.camera_keys
        self.select_state_keys = state_keys or [x for x in self.meta.features if x.startswith(OBS_STATE)]
        self.select_action_keys = action_keys or [x for x in self.meta.features if x.startswith(ACTION)]
        self.select_feature_keys = self.select_video_keys + self.select_state_keys + self.select_action_keys
        self.select_action_is_pad_keys = [f"{k}_is_pad" for k in self.select_action_keys]

    def set_state_relative_action(
        self,
        state_relative_action: bool,
    ):
        """Configure state-relative action transformation (GR00T-style).

        Computes ``action[t:t+h] - state[t]`` for masked dims at training time.
        Requires ``meta/state_relative_norm_stats.json`` to exist in the dataset root
        (generate it with ``tool/compute_state_relative_norm_stats.py``).

        Must be called before ``set_normalization`` so the normalizer is built
        with state-relative statistics.

        Args:
            state_relative_action: Whether to enable state-relative transform.
            action_mask: Per-dim boolean mask (over the concatenated action vector).
                If None, resolved from ``self.embodiment_tag`` → EMBODIMENT_CONFIGS,
                falling back to all-True (all dims state-relative).
        """
        self.state_relative_action = state_relative_action
        self.state_relative_mask = None  # cleared unless enabled

        if not state_relative_action:
            return
        
        if self.embodiment_tag is None:
            self.state_relative_action = False
            cprint(
                f"[Local Rank {self.rank}]: WARNING - embodiment_tag is not set for {self.repo_id} thus state-relative action is disabled. ",
                msg_type="warning",
            )
            return

        # ── Resolve mask ──────────────────────────────────────────────────
        ecfg = EMBODIMENT_CONFIGS.get(self.embodiment_tag)
        if ecfg is not None:
            resolved_mask = ecfg.delta_action_mask

        # Total action dim = sum of all selected action key dims
        total_action_dim = sum(
            self.meta.features[k]["shape"][0] for k in self.select_action_keys
        )

        assert total_action_dim == len(resolved_mask)

        if resolved_mask is not None:
            mask_list = resolved_mask
        else:
            # Default: all action dims are state-relative
            mask_list = [True] * total_action_dim

        self.state_relative_mask = torch.tensor(mask_list, dtype=torch.bool)

        # ── Load state-relative norm stats ────────────────────────────────
        chunk_size = len(self.delta_indices[self.select_action_keys[0]])
        state_rel_stats_path = self.root / "meta" / f"state_relative_ck{chunk_size}_norm_stats.json"
        if not state_rel_stats_path.exists():
            raise FileNotFoundError(
                f"State-relative norm stats not found at '{state_rel_stats_path}'. "
                f"Please generate them first:\n"
                f"  python tool/compute_state_relative_norm_stats.py \\\n"
                f"    --dataset_path {self.root} \\\n"
                f"    --action_keys {' '.join(self.select_action_keys)} \\\n"
                f"    --state_keys {' '.join(self.select_state_keys)}"
            )

        with open(state_rel_stats_path) as f:
            rel_stats_raw = json.load(f)

        # Override meta.stats for action keys with state-relative statistics
        for key in self.select_action_keys:
            if key not in rel_stats_raw:
                cprint(
                    f"[Local Rank {self.rank}]: WARNING - key '{key}' not found in "
                    f"state_relative_norm_stats.json for {self.repo_id}. "
                    f"Falling back to absolute stats.",
                    msg_type="warning",
                )
                continue
            for stat_name, stat_val in rel_stats_raw[key].items():
                if stat_val is not None:
                    self.meta.stats[key][stat_name] = np.array(stat_val, dtype=np.float32)

        cprint(
            f"[Local Rank {self.rank}]: * state-relative action enabled for {self.repo_id} "
            f"(mask={self.state_relative_mask.tolist()})",
            msg_type="info",
        )

    def set_normalization(self, state_mode: str = "MEAN_STD"):
        """set normalization mode for lerobot dataset
        Tips: in PRTS, STATE uses QUANTILE normalization for discrete state tokenization (similar to PI05)
        ACTION also uses the specified state_mode (typically MEAN_STD)
        """
        features = dataset_to_policy_features(self._features)
        mapping = {"STATE": NormalizationMode("QUANTILE"), "ACTION": NormalizationMode(state_mode)}
        self.normalizer = Normalize(features, mapping, self._stats)

    def set_unnormalization(self, state_mode: str = "MEAN_STD"):
        features = dataset_to_policy_features(self._features)
        mapping = {"ACTION": NormalizationMode(state_mode)}
        self.unnormalizer = Unnormalize(features, mapping, self._stats)

    def get_episode_from_index(self, episodes: list[int] | None = None):
        """
        episodes: list of episode indices
        """
        if episodes is None:
            self.episode_from_index = None
        else:
            self.episode_from_index = {ep_idx: i for i, ep_idx in enumerate(episodes)}

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is None:
            fpaths = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.meta.episodes]
            hf_dataset = load_dataset("parquet", data_files=fpaths, split="train", keep_in_memory=False, streaming=False)
        else:
            files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train", keep_in_memory=False, streaming=False)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.select_video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: torch.stack(self.hf_dataset[q_idx][key])
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def __len__(self):
        return self.num_frames

    def _get_query_indices(
        self, idx: int, ep_idx: int, delta_indices: dict = None
    ) -> tuple[dict[str, list[int | bool]]]:
        if self.episode_from_index is not None:
            ep_idx = self.episode_from_index[ep_idx]
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]

        steps_to_go = ep_end - 1 - idx

        delta_indices = delta_indices or self.delta_indices
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in delta_indices.items()  # {"action": [0-50)}
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in delta_indices.items()
        }
        return query_indices, padding, steps_to_go

    def __getitem__(self, idx, delta_indices: dict = None) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding, steps_to_go = self._get_query_indices(idx, ep_idx, delta_indices)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            for cam in self.select_video_keys:
                original_frame = item[cam]
                item[cam] = self.image_transforms(original_frame)
                
                if hasattr(original_frame, 'close'):
                    try:
                        original_frame.close()
                    except:
                        pass
                del original_frame

        task_idx = item["task_index"].item()
        if self.train_subtask and len(self.task_sizes[ep_idx]) > 0:
            try:
                if self.train_subtask == "cumulate":
                    action_config = self.meta.episodes[ep_idx]["action_config"]
                    if isinstance(action_config, dict) and "steps" in action_config:
                        task_text = " ".join(
                            [item["step_description"] for item in action_config["steps"]]
                        )
                    else:
                        task_text = " ".join(
                            [item["action_text"] for item in self.meta.episodes[ep_idx]["action_config"]]
                        )
                else:
                    global_steps_to_go = steps_to_go.clone()
                    # Here we may overwrite the step_to_go, because we would use the subtask as the instruction
                    sub_idx = bisect.bisect_right(self.task_sizes[ep_idx], item["frame_index"])
                    sub_idx = min(sub_idx, len(self.task_sizes[ep_idx]) - 1)

                    steps_to_go = self.task_sizes[ep_idx][sub_idx] - item["frame_index"]
                    steps_to_go = max(0, steps_to_go)

                    action_config = self.meta.episodes[ep_idx]["action_config"]
                    if isinstance(action_config, dict) and "steps" in action_config:
                        task_text = action_config["steps"][sub_idx]["step_description"]
                    else:   
                        task_text = action_config[sub_idx]["action_text"]

                    if isinstance(self.train_subtask, str) and self.train_subtask.startswith("mixture"):
                        global_text = self.meta.tasks[task_idx]
                        w = float(self.train_subtask.split(":")[-1])
                        # random select from [global_text, subtask_text]
                        selected_text = random.choices([task_text, global_text], weights=[w, 1 - w])[0]
                        
                        # Use global_steps_to_go if global_text is selected
                        if selected_text == global_text:
                            steps_to_go = global_steps_to_go
                        
                        task_text = selected_text

            except Exception as e:
                cprint(f'[Local Rank {self.rank}]: {self.repo_id} failed to get subtask {idx} / {len(self.hf_dataset)}: {e} ', msg_type='warning')
                task_text = self.meta.tasks[task_idx]

        else:
            task_text = self.meta.tasks[task_idx]

        task_text = (
                task_text 
                or self.meta.episodes[ep_idx]["tasks"][0] 
                or "Analyze the visual observations and proprioception to execute the intended manipulation task."
            )
        
        item["task"] = task_text.lower().strip().rstrip('.').capitalize() + '.'
        item["cam_names"] = [cam.split(".")[-1] for cam in self.select_video_keys]

        item["steps_to_go"] = steps_to_go

        return self.post_process(item)

    def __getitem_ep_frame__(self, direct_ep_idx, frame_idx, delta_indices: dict = None) -> dict:
        # Use episode idx and frame idx import data
        ep_start = self.episode_data_index["from"][direct_ep_idx]
        idx = ep_start + frame_idx
        item = self.hf_dataset[idx.item()]
        ep_idx = item["episode_index"].item()
        assert(direct_ep_idx == ep_idx)
        ### The rest part stays the same
        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding, steps_to_go = self._get_query_indices(idx, ep_idx, delta_indices)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            for cam in self.select_video_keys:
                item[cam] = self.image_transforms(item[cam])

        task_idx = item["task_index"].item()
        if self.train_subtask and len(self.task_sizes[ep_idx]) > 0:
            try:
                if self.train_subtask == "cumulate":
                    action_config = self.meta.episodes[ep_idx]["action_config"]
                    if isinstance(action_config, dict) and "steps" in action_config:
                        task_text = " ".join(
                            [item["step_description"] for item in action_config["steps"]]
                        )
                    else:
                        task_text = " ".join(
                            [item["action_text"] for item in self.meta.episodes[ep_idx]["action_config"]]
                        )
                else:
                    sub_idx = bisect.bisect_right(self.task_sizes[ep_idx], item["frame_index"])
                    sub_idx = min(sub_idx, len(self.task_sizes[ep_idx]) - 1)
                    action_config = self.meta.episodes[ep_idx]["action_config"]
                    if isinstance(action_config, dict) and "steps" in action_config:
                        task_text = action_config["steps"][sub_idx]["step_description"]
                    else:   
                        task_text = action_config[sub_idx]["action_text"]
                        
                    if isinstance(self.train_subtask, str) and self.train_subtask.startswith("mixture"):
                        global_text = self.meta.tasks[task_idx]
                        w = float(self.train_subtask.split(":")[-1])
                        # random select from [global_text, subtask_text]
                        task_text = random.choices([task_text, global_text], weights=[w, 1 - w])[0]

            except Exception as e:
                cprint( f'[Local Rank {self.rank}]: {self.repo_id} failed to get subtask {idx} / {len(self.hf_dataset)}: {e} ', msg_type='warning')
                task_text = self.meta.tasks[task_idx]

        else:
            task_text = self.meta.tasks[task_idx]
        
        task_text = (
                task_text 
                or self.meta.episodes[ep_idx]["tasks"][0] 
                or "Analyze the visual observations and proprioception to execute the intended manipulation task."
            )

        item["task"] = task_text.lower().strip().rstrip('.').capitalize() + '.'
        item["cam_names"] = [cam.split(".")[-1] for cam in self.select_video_keys]

        item["steps_to_go"] = steps_to_go

        return self.post_process(item)
    
    @property
    def _stats(self) -> datasets.Features:
        return {k: self.meta.stats[k] for k in (self.select_state_keys + self.select_action_keys)}

    @property
    def _features(self) -> dict[str, dict]:
        return {k: self.meta.features[k] for k in self.select_feature_keys}

    def _apply_state_relative_transform(self, item: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Subtract current state from action for masked dims (GR00T-style state-relative).

        Transform applied in-place on a shallow copy of item:
            action_chunk[h, :mask_dims] -= state[:mask_dims]   (where mask[dim] is True)

        Requires ``observation.state`` to be present, which is the concatenation of all
        ``select_state_keys`` in the item (standard LeRobot convention).

        The mask is ``self.state_relative_mask`` (shape: [total_action_dim]).
        """
        mask = self.state_relative_mask  # (total_action_dim,)
        mask_dims = mask.shape[0]

        # Build concatenated state reference: shape (total_state_dim,)
        state_parts = [item[k].float() for k in self.select_state_keys if k in item]
        if not state_parts:
            return item
        state_ref = torch.cat(state_parts, dim=-1)  # (total_state_dim,)

        # Expand state to broadcast over the action horizon:  (1, total_state_dim)
        state_expanded = state_ref.unsqueeze(0)  # will broadcast over (H, dim)

        # Process each action key and accumulate transformed slices
        dims_done = 0
        for key in self.select_action_keys:
            if key not in item:
                continue
            action = item[key].float()  # (H, key_action_dim) or (key_action_dim,)
            key_dim = action.shape[-1]

            # Which part of the global mask covers this key's dims?
            key_mask_start = dims_done
            key_mask_end = min(dims_done + key_dim, mask_dims)
            dims_done += key_dim

            if key_mask_start >= mask_dims:
                # This key's dims are fully beyond the mask → all absolute, skip
                continue

            # Slice of the global mask that corresponds to this key
            key_mask = mask[key_mask_start:key_mask_end]  # (<=key_dim,)
            covered = key_mask.shape[0]  # number of dims in this key that are covered

            # Slice of state that aligns with this key's covered dims
            state_slice = state_expanded[..., key_mask_start:key_mask_end]  # (1, covered)

            # Apply subtraction only for True dims
            action_copy = action.clone()
            action_copy[..., :covered] = action[..., :covered] - torch.where(
                key_mask,
                state_slice.expand_as(action[..., :covered]),
                torch.zeros_like(action[..., :covered]),
            )
            item[key] = action_copy

        return item

    def post_process(self, item: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """sort the keys in the order of select_feature_keys"""
        item = {k: item[k] for k in (self.select_feature_keys + ["task"] + self.select_action_is_pad_keys + ["cam_names"] + ["steps_to_go"])}

        # State-relative transform: action[t+h] - state[t] for masked dims.
        # Must happen BEFORE normalization (normalizer uses state-relative stats).
        if self.state_relative_action:
            item = self._apply_state_relative_transform(item)

        # Add quantile obs_state and action
        if self.load_quantile_stats:
            for key in self.select_action_keys:
                q01, q99 = self.meta.stats[key]['q01'][..., : item[key].shape[-1]], self.meta.stats[key]['q99'][..., : item[key].shape[-1]]

                result = (item[key].to(torch.float32) - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
                result = torch.clamp(result, min=-50.0, max=50.0)

                item[f"quantile_normalized_{key}"] = result

        item = self.normalizer(item)
        return item


class MultiLeRobotDataset(BaseMultiLeRobotDataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`.
    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        data_configs: list[LerobotConfig],
        state_mode: str = "MEAN_STD",
        image_transforms: Callable | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        chunk_size: int = 32,
        rank: int = 0,
        **kwargs: Any,
    ):
        self.data_configs = data_configs
        self.chunk_size = chunk_size

        # load lerobot datasets with timeout handling
        num_processes = int(os.environ.get("DATASET_NUM_PROCESSES", 8))
        timeout_seconds = int(os.environ.get("DATASET_LOAD_TIMEOUT", 1200))  # 800 s default
        # 修改1: 添加批次大小参数，控制每批加载的数据集数量
        batch_size = int(os.environ.get("DATASET_BATCH_SIZE", 700))  # 默认每批700个数据集
        
        cprint(f"[Local Rank {rank}]: * load {len(data_configs)} lerobot datasets with {num_processes} processes (timeout: {timeout_seconds}s, batch_size: {batch_size}) ...", msg_type='info')

        # 修改2: 按批次处理数据集
        datasets = {}
        all_indices = list(range(len(data_configs)))
        start_time = time.time()

        # 修改3: 将数据集分成多个批次
        for batch_start in range(0, len(all_indices), batch_size):
            batch_indices = all_indices[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(all_indices) + batch_size - 1) // batch_size
            
            cprint(f"[Local Rank {rank}]: * Processing batch {batch_num}/{total_batches} (datasets {batch_start}-{min(batch_start+batch_size-1, len(all_indices)-1)})", msg_type='info')
            
            # 修改4: 仅对当前批次使用并行加载
            remaining_indices = batch_indices.copy()
            use_multiprocessing = True
            current_time = time.time()
            batch_start_time = current_time

            if num_processes > 1 and len(batch_indices) > 1:
                try:
                    ctx = multiprocessing.get_context("spawn")
                    async_results = {}
                    loaded_count = 0
                    
                    fn = partial(
                        _load_single_lerobot_dataset,
                        data_configs=data_configs,
                        image_transforms=image_transforms,
                        download_videos=download_videos,
                        video_backend=video_backend,
                        chunk_size=chunk_size,
                        state_mode=state_mode,
                        rank=rank
                    )
                    
                    try:
                        with ctx.Pool(processes=min(num_processes, len(batch_indices))) as pool:
                            for idx in remaining_indices:
                                async_results[idx] = pool.apply_async(fn, (idx,))
                            
                            updated = False
                            while int(current_time - batch_start_time) <= timeout_seconds and loaded_count < len(batch_indices):
                                for idx in list(remaining_indices):
                                    if idx not in async_results:
                                        continue
                                    async_result = async_results[idx]
                                    if not async_result.ready():
                                        continue
                                    try:
                                        result = async_result.get(timeout=5)
                                        if result is not None:
                                            datasets[idx] = result
                                            loaded_count += 1
                                            remaining_indices.remove(idx)
                                            del async_results[idx]
                                            updated = True
                                    except Exception as e:
                                        cprint(f"[Local Rank {rank}] Dataset {idx} failed during async loading: {str(e)} ", msg_type='fail')
                                        remaining_indices.remove(idx)
                                        del async_results[idx]
                                if updated:
                                    cprint(f"[Local Rank {rank}] Batch {batch_num}: Dataset {loaded_count}/{len(batch_indices)} loaded successfully! ", msg_type='success')
                                    updated = False
                                time.sleep(2)  
                                current_time = time.time()
                    
                    except Exception as e:
                        cprint(f"[Local Rank {rank}] Pool operation failed: {str(e)}. ", msg_type='fail')
                    
                    async_results.clear()
                    
                except Exception as e:
                    cprint(f"[Local Rank {rank}] Multiprocessing failed: {str(e)}. Falling back to sequential loading. ", msg_type='fail')
                    use_multiprocessing = False

            # Fallback to sequential loading for remaining datasets in this batch
            if remaining_indices:
                if not use_multiprocessing:
                    cprint(f"[Local Rank {rank}]: * Falling back to sequential loading for {len(remaining_indices)} remaining datasets in batch {batch_num}...", msg_type='info')
                
                for idx in tqdm(remaining_indices.copy(), desc=f"Loading batch {batch_num} remaining datasets"):
                    try:
                        ds = _load_single_lerobot_dataset(
                            idx,
                            data_configs=data_configs,
                            image_transforms=image_transforms,
                            download_videos=download_videos,
                            video_backend=video_backend,
                            chunk_size=chunk_size,
                            state_mode=state_mode,
                            rank=rank
                        )
                        if ds is not None:
                            datasets[idx] = ds
                        if idx in remaining_indices:
                            remaining_indices.remove(idx)
                    except Exception as e:
                        cprint(f"[Local Rank {rank}] Dataset {idx} failed during sequential loading: {str(e)} ", msg_type='fail')
                        if idx in remaining_indices:
                            remaining_indices.remove(idx)
            
            # 修改5: 每个批次完成后立即进行内存回收
            batch_time = time.time() - batch_start_time
            cprint(f"[Local Rank {rank}]: * Batch {batch_num}/{total_batches} completed in {batch_time:.2f} seconds. Performing memory cleanup...", msg_type='info')
            import gc
            gc.collect()
            torch.cuda.empty_cache() if hasattr(torch, 'cuda') and torch.cuda.is_available() else None

        # 保持原有代码不变
        self._datasets = [datasets[i] for i in range(len(data_configs)) if i in datasets]
        self.repo_ids = [ds.repo_id for ds in self._datasets]
        current_time = time.time()
        cprint(f"[Local Rank {rank}] Successfully load dataset {len(self.repo_ids)}/{len(data_configs)}:\n{self.repo_ids} with {current_time-start_time} seconds! ", msg_type='success')

        self._repo_ids_index = {repo_id: i for i, repo_id in enumerate(self.repo_ids)}
        self.cumulative_sizes = ConcatDataset.cumsum(self._datasets)
        self.image_transforms = image_transforms

        # set select feature keys
        self.state_mode = state_mode
        self._select_video_keys = {
            ds.repo_id.replace("/", "."): ds.select_video_keys for ds in self._datasets
        }
        self._select_state_keys = {
            ds.repo_id.replace("/", "."): ds.select_state_keys for ds in self._datasets
        }
        self._select_action_keys = {
            ds.repo_id.replace("/", "."): ds.select_action_keys for ds in self._datasets
        }
        
        del datasets
        import gc
        gc.collect()

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        item = self._datasets[dataset_idx][sample_idx]

        item['dataset_names'] = self.repo_ids[dataset_idx]

        return self.post_process(item)

    def getitem_by_ep_frame(self, repo_id: str, ep_idx: int, frame_idx: int) -> dict[str, torch.Tensor]:
        """get an item by repo_id, episode idx and frame idx."""
        dataset_idx = self._repo_ids_index.get(repo_id)
        if dataset_idx is None:
            raise ValueError(f"\033[31m ❌ Invalid dataset: {repo_id}. Available dataset: {self.repo_ids} \033[0m")
        lerobot_dataset = self._datasets[dataset_idx]

        delta_indices = {k: list(range(0, self.chunk_size)) for k in lerobot_dataset.select_action_keys}

        item = lerobot_dataset.__getitem_ep_frame__(ep_idx, frame_idx, delta_indices=delta_indices)
        return self.post_process(item)
    
    def post_process(self, item: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """unify video keys across datasets."""
        return item

    @property
    def _features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            repo_id_suffix = dataset.repo_id.replace("/", ".")
            features[repo_id_suffix] = dataset._features
        return features

    @property
    def _stats(self) -> datasets.Features:
        stats = {}
        for dataset in self._datasets:
            repo_id_suffix = dataset.repo_id.replace("/", ".")
            stats[repo_id_suffix] = dataset._stats
        return stats

    @property
    def configuration(self) -> dict:
        return {
            "features": self._features,
            "stats": serialize_dict(self._stats),
            "state_mode": self.state_mode,
            "select_video_keys": self._select_video_keys,
            "select_state_keys": self._select_state_keys,
            "select_action_keys": self._select_action_keys,
        }


def _load_single_lerobot_dataset(
    idx,
    data_configs: list[LerobotConfig],
    image_transforms: Callable | None = None,
    download_videos: bool = True,
    video_backend: str | None = None,
    chunk_size: int = 32,
    state_mode: str = None,
    rank: int = 0
):
    """load a single lerobot dataset"""
    try:
        data_config = data_configs[idx]
        data_path = Path(data_config.root) if data_config.root else HF_LEROBOT_HOME / data_config.repo_id
        # data_path = Path(data_config.root or HF_LEROBOT_HOME) / data_config.repo_id
        meta = LeRobotDatasetMetadata(data_config.repo_id, data_path)
        select_action_keys = data_config.select_action_keys or [
            k for k in meta.features if k.startswith(ACTION)
        ]
        delta_timestamps = {k: [i / meta.fps for i in range(0, chunk_size)] for k in select_action_keys}
        dataset = LeRobotDataset(
            data_config.repo_id,
            root=data_path,
            episodes=data_config.episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            download_videos=download_videos,
            video_backend=video_backend,
            select_video_keys=data_config.select_video_keys,
            select_state_keys=data_config.select_state_keys,
            select_action_keys=data_config.select_action_keys,
            state_mode=state_mode,  # data_config.state_mode,
            train_subtask=data_config.train_subtask,
            load_quantile_stats=data_config.load_quantile_stats,
            state_relative_action=data_config.state_relative_action,
            embodiment_tag=data_config.embodiment_tag,
            # weight=data_config.weight,
            rank=rank
        )
        cprint(f"[Local Rank {rank}]: Successfully load dataset {data_config.repo_id} !", msg_type='success')
    except Exception as e:
        cprint(f"[Local Rank {rank}] Read dataset {data_config.repo_id} failed: {e}, skipped! ", msg_type='warning')
        return None
    return dataset
