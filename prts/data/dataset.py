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
import random
import logging
import hashlib
import torch
import itertools
import transformers
import numpy as np
from lerobot.constants import ACTION, OBS_IMAGE, OBS_STATE
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence
from transformers.data.data_collator import DefaultDataCollator
from typing import Dict, Optional, Sequence, List, Tuple, Any
import torch.distributed as dist

from .config import initialize_collections
from .rope2d import get_rope_index_3, get_rope_index_25, get_rope_index_2
from .lerobot_dataset import MultiLeRobotDataset
from .multim_dataset import MultimodalDataset, pad_vector
from .schema import DataConfig, LerobotConfig
from .transforms import ImageTransforms, ImageTransformsConfig
from ..training.config import TrainConfig
from ..training.train_utils import (
    to_list_fast,
    aggregate_dataset_length,   
    validate_packing_efficiency,
    pack_data_points_by_length_with_WFD,
    pack_data_points_by_length_with_shuffle,
    pack_data_points_by_length_with_shuffle_streaming,
    pack_data_points_by_length_with_shuffle_ultra_memory_efficient,
    EfficientLengthStorage,
    LeRobotStylePacker
)
from ..constants import (
    # for discrete action token
    ACTION_START_TOKEN,
    ACTION_END_TOKEN,
    ACTION_PLACEHOLDER_TOKEN,
    # CRL repr special tokens
    CRL_GOAL_REPR_TOKEN,
    CRL_OBS_REPR_TOKEN,
    # qwen2.5-vl special tokens
    IM_START_TOKEN,
    IM_END_TOKEN,
    IMAGE_PLACEHOLDER_TOKEN,
    VIDEO_PLACEHOLDER_TOKEN,
    VISION_START_TOKEN,
    VISION_END_TOKEN,
    PAD_TOKEN,
    SYSTEM_MESSAGE,
    IGNORE_INDEX,
)
from prts.utils.logging_utils import cprint

# set the local rank for distributed training
local_rank = dist.get_rank() if dist.is_initialized() else 0

def rank0_print(message, msg_type='normal', *args):
    if local_rank == 0:
        cprint(message=message, msg_type=msg_type, *args)


def instruction_to_task_id(instruction: str) -> int:
    """Deterministic hash of normalized instruction text → int64 task ID.

    Used for false-negative masking in CRL: samples with the same task_id
    are excluded from the set of negatives during InfoNCE computation.
    """
    normalized = instruction.strip().lower()
    h = hashlib.md5(normalized.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=True)


def find_subsequence_cpu_fastest(input_ids: torch.Tensor, subsequence: list[int]) -> int | None:
    # 1. 零拷贝转换为 Numpy (前提是 tensor 在 CPU 且连续)
    # 如果 input_ids 经过了切片操作，建议先 .contiguous()
    arr_np = input_ids.numpy()
    
    # 2. 准备子序列的 Numpy 数组，确保数据类型完全一致（例如都是 int64）
    sub_np = np.array(subsequence, dtype=arr_np.dtype)
    
    # 3. 转换为字节流
    arr_bytes = arr_np.tobytes()
    sub_bytes = sub_np.tobytes()
    
    # 4. 使用 Python 内置的高速字符串查找
    byte_index = arr_bytes.find(sub_bytes)
    
    if byte_index == -1:
        return None
        
    # 5. 将字节偏移量转换为元素索引
    # itemsize 是每个元素的字节数（int64 是 8）
    return byte_index // arr_np.itemsize


def compute_crl_spans(
    input_ids: torch.Tensor,
    instruction_marker_ids: List[int],
    predict_marker_ids: List[int],
    action_start_token_id: int,
    action_end_token_id: int,
    goal_repr_token_id: Optional[int] = None,
) -> Optional[Dict[str, int]]:
    """
    Compute CRL-related token spans for a single sample.
    
    Args:
        input_ids: 1D tensor of token IDs
        instruction_marker_ids: List of token IDs for "Instruction:"
        predict_marker_ids: List of token IDs for ".\nPredict" (fallback when goal_repr_token_id is None)
        action_start_token_id: Token ID for <|action_start|>
        action_end_token_id: Token ID for <|action_end|>
        goal_repr_token_id: Token ID for <|goal_repr|>. When present in the BC sequence,
            used as the instruction end marker (more robust than subsequence matching).
        
    Returns:
        Dictionary with instruction_start, instruction_end, action_start, action_end, goal_pos
        or None if spans cannot be determined
    """
    # Find instruction start (after "Instruction:")
    instr_start = find_subsequence_cpu_fastest(input_ids.contiguous(), instruction_marker_ids)

    if instr_start is None:
        return None

    # Find instruction end / goal position
    if goal_repr_token_id is not None:
        # Use <|goal_repr|> token as instruction end marker
        goal_repr_positions = (input_ids == goal_repr_token_id).nonzero(as_tuple=True)[0]
        if len(goal_repr_positions) == 0 or goal_repr_positions[0].item() <= instr_start:
            return None
        goal_pos = goal_repr_positions[0].item()
        instr_end = goal_pos + 1
    else:
        # Fallback: find instruction end via ".\nPredict" subsequence
        instr_end = find_subsequence_cpu_fastest(input_ids, predict_marker_ids)
        if instr_end is None or instr_end <= instr_start:
            return None
        instr_end += 1
        goal_pos = instr_end - 1

    if goal_pos < instr_start:
        return None
    
    # Find action span
    action_start_positions = (input_ids == action_start_token_id).nonzero(as_tuple=True)[0]
    action_end_positions = (input_ids == action_end_token_id).nonzero(as_tuple=True)[0]
    
    if len(action_start_positions) == 0 or len(action_end_positions) == 0:
        return None
    
    action_start = action_start_positions[0].item()
    action_end = action_end_positions[0].item()
    
    return {
        "instruction_start": instr_start,
        "instruction_end": instr_end,
        "action_start": action_start,
        "action_end_pos": action_end,
        "goal_pos": goal_pos,
    }

"""multimodal lerobot datasets"""
class MultimodalLeRobotDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        args: TrainConfig,
        processor: transformers.ProcessorMixin,
        padding=True,       # 这里始终默认为True
    ):
        super().__init__()
        self.args = args

        if args.rope_type == "qwen3_vl":
            self.get_rope_index = get_rope_index_3
            cprint(f"Using Qwen3VL rope index", msg_type='info')
        elif args.rope_type == "qwen2_5_vl":
            self.get_rope_index = get_rope_index_25
            cprint(f"Using Qwen2.5VL rope index", msg_type='info')
        else:
            raise ValueError(f"\033[1;33m ⚠️ model_type: {args.rope_type} not supported \033[0m")

        mm_dataset = []
        lerobot_dataset = []

        assert not (args.train_lerobot_only and args.train_mm_only)

        ### 这里只有在使用yaml的情况下才会混合mm dataset和lerobot dataset一起训练
        if args.data_path.endswith(".yaml"):
            data_configs = DataConfig.from_yaml(args.data_path)
            if args.train_lerobot_only:
                data_configs.mm_datasets = []
        else:
            data_configs = DataConfig(
                lerobot_datasets=[LerobotConfig(repo_id=args.data_path)],
                mm_datasets=[],
            )

        # load lerobot datasets
        if len(data_configs.lerobot_datasets) > 0:
            lerobot_dataset = MultiLeRobotDataset(
                data_configs=data_configs.lerobot_datasets,
                image_transforms=ImageTransforms(ImageTransformsConfig()),
                video_backend=args.lerobot_data_video_backend,
                state_mode=args.state_mode,
                chunk_size=args.chunk_size,
                rank=local_rank
            )

        # load mm datasets
        if len(data_configs.mm_datasets) > 0:
            mm_dataset = MultimodalDataset(
                data_configs=data_configs.mm_datasets,
                max_num_tokens=args.max_packed_length if args.pack_dataset else 5120,
                meta_dataset=lerobot_dataset,
                max_action_dim=args.max_action_dim,
                chunk_size=args.chunk_size,
                phase=args.training_phase, # "pre-training"
                rank=local_rank
            )

        # multi-modal datasets
        self.mm_dataset = mm_dataset
        self.lerobot_dataset = lerobot_dataset

        self.fps = args.fps
        self.padding = padding
        self.image_min_pixel = args.image_min_pixels
        self.image_max_pixel = args.image_max_pixels
        self.video_min_pixel = args.video_min_pixels
        self.video_max_pixel = args.video_max_pixels
        self.image_resized_w = args.image_resized_width
        self.image_resized_h = args.image_resized_height
        self.video_resized_w = args.video_resized_width
        self.video_resized_h = args.video_resized_height
        self.vision_base_paths = self.mm_dataset.vision_base_paths if mm_dataset else None
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)

        rank0_print("Formatting inputs...Skip in lazy mode", msg_type='info')
        processor = update_processor_pixels(processor, args)
        self.processor = processor

        # decide whether to convert fast action tokens or not
        self.train_action_expert = args.train_action_expert
        self.ratio_fast_action_tokens = args.ratio_fast_action_tokens
        self.use_crl_sample = False
        self.gamma = args.gamma
        self.use_multi_positive = False

        ### For fast crl data construction
        self.action_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(ACTION_START_TOKEN)
        self.action_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(ACTION_END_TOKEN)
        self.instruction_marker_ids = self.processor.tokenizer.encode("Instruction:", add_special_tokens=False)
        self.predict_marker_ids = self.processor.tokenizer.encode(".\nPredict", add_special_tokens=False)
        self.crl_goal_repr_token_id = self.processor.tokenizer.convert_tokens_to_ids(CRL_GOAL_REPR_TOKEN)
        self.crl_obs_repr_token_id = self.processor.tokenizer.convert_tokens_to_ids(CRL_OBS_REPR_TOKEN)
    
    @property
    def lengths(self):
        """group the lengths of the datasets, we set sample_actions to False \
            to avoid action sampling damaging the length of the dataset
            After the length is calculated, reset it back to True
        """
        if getattr(self, "cached_lengths", None):
            return self.cached_lengths
        return []

    def __len__(self):
        if self.args.train_mm_only:
            return len(self.mm_dataset)
        else:
            return len(self.mm_dataset) + len(self.lerobot_dataset)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and not getattr(self, "_worker_gc_enabled", False):
            import gc
            gc.enable() 
            self._worker_gc_enabled = True
        _temp_pil_images = []

        if i < len(self.mm_dataset):
            sources = self.mm_dataset[i]
            is_action_data = False
        else:
            is_action_data = True
            item = self.lerobot_dataset[i - len(self.mm_dataset)]
            images, actions, states, quantile_normalized_actions = [], [], [], []
            for k, v in item.items():
                if k.startswith(OBS_IMAGE):
                    images.append(v)
                elif k.startswith(ACTION) and "is_pad" not in k:
                    actions.append(v.unsqueeze(-1) if v.dim() == 1 else v)
                elif k.startswith(OBS_STATE):
                    states.append(v)
                elif k.startswith(ACTION) and "is_pad" in k:
                    action_is_pad = v
                elif k.startswith("quantile_normalized_") and k[len("quantile_normalized_"):] in item:
                    quantile_normalized_actions.append(v.unsqueeze(-1) if v.dim() == 1 else v)
            
            # states, state_dof_mask = pad_vector(torch.cat(states, dim=-1), self.args.max_action_dim)
            states = torch.cat(states, dim=-1)
            actions, action_dof_mask = pad_vector(torch.cat(actions, dim=-1), self.args.max_action_dim)
            quantile_normalized_actions = torch.cat(quantile_normalized_actions, dim=-1)

            # for contrastive RL
            step_to_go = item.get("steps_to_go", 0)

            ### design prompt (modified by Yang Zhang)
            camera_names = item.get('cam_names', ['view ' + str(j + 1) for j in range(len(images))])
            cam_mapping = {
                "image_0": "camera_top",
                "image_1": "camera_left_wrist",
                "image_2": "camera_right_wrist"
            }
            new_camera_names = []
            for idx, orig_name in enumerate(camera_names):
                if orig_name in cam_mapping:
                    new_camera_names.append(cam_mapping[orig_name])
                elif "image" in orig_name:
                    new_camera_names.append(orig_name.replace("image", "camera"))
                elif "cam" in orig_name:
                    new_camera_names.append(orig_name.replace("cam", "camera"))
                else:
                    new_camera_names.append("camera_" + orig_name)
            
            camera_names = new_camera_names
            camera_names = [name.replace('_', ' ') for name in camera_names]
            camera_names = [name.capitalize() for name in camera_names]

            image_replacements = []
            for cam_name in camera_names:
                token_str = f"{cam_name}: {VISION_START_TOKEN}{IMAGE_PLACEHOLDER_TOKEN}{VISION_END_TOKEN}\n"
                image_replacements.append(token_str)

            image_block = "".join(image_replacements)

            # Discretize state to discrete bins (similar to PI05)
            # States are already normalized to [-1, 1] range via MIN_MAX normalization
            if len(states) > 0:
                # 另一种方式我认为可以像QwenVL里面一样将state normalized 到 [0, 1] 的范围，然后 * 1000, 将其归一化到[0, 999]，然后变成string，然后拼接在prompt里面
                discretized_state = ((states + 1) / 2 * 1000).to(torch.int32)
                # 由于使用的是quantile normalization，所以会有边界，对于边界值(异常值)，我们设置其为 -1
                discretized_state = torch.clamp(discretized_state, -1, 1000)
                state_str = " ".join(map(str, discretized_state.tolist()))

                # Replace placeholder token with discretized state string
                state_block = f"Proprioception (normalized to 0-1000 scale): {state_str}\n"
            else:
                # Fallback if no states available
                state_block = f""

            ## final content
            obs_repr_marker = CRL_OBS_REPR_TOKEN if self.use_crl_sample else ""
            goal_repr_marker = CRL_GOAL_REPR_TOKEN if self.use_crl_sample else ""
            user_content = f"{image_block}{state_block}{obs_repr_marker}Instruction: {item['task']}{goal_repr_marker}\nPredict the next action chunk in low-level robotics action format."

            ### ----------------------
          
            use_fast_action_token = False if self.train_action_expert else True

            sources = {
                "conversations": [
                    {
                        "role": "user",
                        "content": user_content,
                    },
                    {
                        "role": "assistant",
                        "content": f"{ACTION_START_TOKEN}{ACTION_PLACEHOLDER_TOKEN}{ACTION_END_TOKEN}" if use_fast_action_token \
                            else "",
                    },
                ],
                "action": [actions], # [] if use_fast_action_token else [actions],
                "quantile_normalized_action": [quantile_normalized_actions] if use_fast_action_token else [],
                "image": images,
                "action_is_pad": [action_is_pad], # [] if use_fast_action_token else [action_is_pad],
                "action_dof_mask": [action_dof_mask], # [] if use_fast_action_token else [action_dof_mask],
                "dataset_names": item['dataset_names']
            }
        
        ## image and video preprocessing is absolutely same as the procedure in Qwen2.5-VL
        if "image" in sources:
            videos = None
            image_files = sources["image"]
            if isinstance(image_files, str):
                image_files = [image_files]
            images = []
            for image_file in image_files:
                if isinstance(image_file, str) and not image_file.startswith("http"):
                    image_folder = self.vision_base_paths[sources["vision_base_idx"]]
                    image_file = os.path.join(image_folder, image_file)
                elif isinstance(image_file, torch.Tensor):  # lerobot dataset
                    image_file = Image.fromarray((image_file * 255).to(torch.uint8).permute(1, 2, 0).numpy())
                    _temp_pil_images.append(image_file)
                images.append(
                    get_image_info(
                        image_file,
                        self.image_min_pixel,
                        self.image_max_pixel,
                        self.image_resized_w,
                        self.image_resized_h,
                        model_type=self.args.rope_type,
                    )
                )
        elif "video" in sources:
            images = None
            video_files = sources["video"]
            video_folder = self.vision_base_paths[sources["vision_base_idx"]]
            if isinstance(video_files, str):
                video_files = [video_files]
            videos = []
            video_metadatas_list = []
            for video_file in video_files:
                if isinstance(video_file, str) and not video_file.startswith("http"):
                    video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs, video_metadatas = get_video_info(
                    video_file,
                    self.video_min_pixel,
                    self.video_max_pixel,
                    self.video_resized_w,
                    self.video_resized_h,
                    self.args.fps,
                    model_type=self.args.rope_type,
                )
                videos.append(video_input)
                video_metadatas_list.append(video_metadatas)
        
        actions = sources.get("action", [])     # 只有在mm dataset中使用lerobot interleaved的sample才会存在多个action在一个list里面，即len(actions) > 1
        states = sources.get("state", [])       # 同上
        quantile_actions = sources.get("quantile_normalized_action", [])
        action_is_pad = sources.get("action_is_pad", [])
        state_dof_mask = sources.get("state_dof_mask")
        action_dof_mask = sources.get("action_dof_mask", [])

        dataset_names = sources.get('dataset_names', 'PRTS_normal')

        conversations = sources["conversations"]

        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_pixel_values_videos =[]
        all_video_grid_thw = []
        all_second_per_grid_ts = []

        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{IM_END_TOKEN}\n"
            system_message_input_ids = self.processor.tokenizer(
                system_message, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX)
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        ## safe checker for the training
        assert len(actions) == len(action_is_pad) == len(action_dof_mask), "Mismatch in action, action_is_pad and action_dof_mask lengths"

        ## 把多轮对话变成batch中的一个sample
        img_start = 0
        action_start = 0
        for _, j in enumerate(range(0, len(conversations), 2)):
            user_input = conversations[j]
            gpt_response = conversations[j + 1]
            user_input = f"{IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{IM_END_TOKEN}\n{IM_START_TOKEN}{gpt_response['role']}\n"
            if IMAGE_PLACEHOLDER_TOKEN in user_input:
                img_num = user_input.count(IMAGE_PLACEHOLDER_TOKEN)
                inputs = self.processor(
                    text=[user_input],
                    images=images[img_start : img_start + img_num] if images else None,
                    videos=videos,
                    padding=False,
                    do_resize=False,
                    return_tensors="pt",
                )
                prompt_input_ids = inputs["input_ids"]
                all_pixel_values.append(inputs['pixel_values'])
                all_image_grid_thw.append(inputs['image_grid_thw'])
                img_start += img_num
            elif VIDEO_PLACEHOLDER_TOKEN in user_input:
                inputs = self.processor(
                    text=[user_input],
                    images=images,
                    videos=videos,
                    padding=False,
                    do_resize=False,
                    return_tensors="pt",
                    video_metadata=video_metadatas_list,
                    **video_kwargs,
                )
                if "second_per_grid_ts" in inputs:
                    all_second_per_grid_ts.append(inputs["second_per_grid_ts"])

                prompt_input_ids = inputs["input_ids"]
                all_pixel_values_videos.append(inputs['pixel_values_videos'])
                all_video_grid_thw.append(inputs['video_grid_thw'])
            else:
                prompt_input_ids = self.processor.tokenizer(
                    user_input, add_special_tokens=False, padding=False, return_tensors="pt"
                )["input_ids"]

            ### processing FAST action token
            current_response_content = gpt_response['content']
            if ACTION_PLACEHOLDER_TOKEN in current_response_content:
                actions_fast_tokens = self.processor.action_tokenizer(
                    # actions[action_start][:, action_dof_mask[action_start][0].bool()]
                    quantile_actions[action_start],
                )
                actions_fast_tokens = [f"<|action_token_{j}|>" for j in actions_fast_tokens[0]]
                actions_fast_token_strs = "".join(actions_fast_tokens)
                current_response_content = current_response_content.replace(
                    ACTION_PLACEHOLDER_TOKEN,
                    actions_fast_token_strs
                )
                action_start += 1

            if len(current_response_content) > 0:
                final_gpt_response = f"{current_response_content}{IM_END_TOKEN}\n"
            else:
                final_gpt_response = f"{current_response_content}"

            # 这里需要利用fast_tokenizer进行action token解算
            response_input_ids = self.processor(text=[final_gpt_response], padding=False, return_tensors="pt")[
                "input_ids"
            ]
            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).long().squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            ).long()

            # ignore the action token. 这里显然我们不会再有这些了
            cached_action_ids = torch.tensor([self.processor.action_token_id])
            action_mask = torch.isin(labels, cached_action_ids)
            assert action_mask.sum() == 0, "The input_ids contain illegal tokens. Please check the code."
            # labels[action_mask] = IGNORE_INDEX

            all_input_ids.append(input_ids)
            all_labels.append(labels)


        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)
        # attention_mask = (input_ids > -1000000).to(torch.long)
        attention_mask = [input_ids.size(0)]

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            input_ids,
            image_grid_thw=torch.cat(all_image_grid_thw, dim=0) if all_image_grid_thw else None,
            video_grid_thw=(
                torch.cat(all_video_grid_thw, dim=0) if all_video_grid_thw else None
            ),
            second_per_grid_ts=torch.cat(all_second_per_grid_ts, dim=0) if all_second_per_grid_ts else None,
        )

        data_dict = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "seq_length": input_ids.shape[0],
            "attention_mask": attention_mask,
            "dataset_names": dataset_names,
            "labels": labels,
        }

        # if not self.args.train_lerobot_only:
        #     data_dict["labels"] = labels

        if len(all_pixel_values) > 0 and len(all_image_grid_thw) > 0:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(all_pixel_values_videos) > 0 and len(all_video_grid_thw) > 0:
            pixel_video_values = torch.cat(all_pixel_values_videos, dim=0)
            video_thw = torch.cat(all_video_grid_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw
            if len(all_second_per_grid_ts) > 0:
                data_dict["second_per_grid_ts"] = torch.cat(all_second_per_grid_ts, dim=0)

        if len(actions) > 0:
            actions = torch.stack(actions, dim=0)
            action_is_pad = torch.stack(action_is_pad, dim=0)
            action_dof_mask = torch.stack(action_dof_mask, dim=0)

            data_dict["actions"] = actions
            data_dict["action_is_pad"] = action_is_pad
            data_dict["action_dof_mask"] = action_dof_mask

        if len(states) > 0:
            states = torch.stack(states, dim=0)
            state_dof_mask = torch.stack(state_dof_mask, dim=0)

            data_dict["states"] = states
            data_dict["state_dof_mask"] = state_dof_mask

        # Compute obs-only RoPE offset for DiT action tokens.
        # Needed whenever train_action_expert is True so that _make_action_rope_cos_sin
        # can assign consistent position ids regardless of whether action tokens appear
        # in the VLM input sequence (training) or not (inference).
        # if self.train_action_expert and is_action_data:
        #     _action_start_pos = (input_ids == self.action_start_token_id).nonzero(as_tuple=True)[0]
        #     _obs_end_idx = int(_action_start_pos[0].item()) if len(_action_start_pos) > 0 else input_ids.shape[0]
        #     _obs_max_pos = position_ids[:, :, :_obs_end_idx].max()
        #     data_dict["dit_obs_offset"] = torch.tensor(_obs_max_pos.item() + 1, dtype=torch.long)  # (1,)

        # Compute CRL spans and extend sequence with CRL action tokens for single-pass training
        # Only compute for lerobot samples which have action tokens
        if self.use_crl_sample and is_action_data:
            crl_span = compute_crl_spans(
                input_ids=input_ids,
                instruction_marker_ids=self.instruction_marker_ids,
                predict_marker_ids=self.predict_marker_ids,
                action_start_token_id=self.action_start_token_id,
                action_end_token_id=self.action_end_token_id,
                goal_repr_token_id=self.crl_goal_repr_token_id,
            )
            if crl_span is not None:
                # --- 1. Append CRL action tokens (sa_repr) ---
                action_start_idx = crl_span["action_start"]
                action_end_idx = crl_span["action_end_pos"]
                crl_tokens = data_dict["input_ids"][action_start_idx : action_end_idx + 1].clone()
                crl_len = crl_tokens.shape[0]
                
                crl_start_pos = data_dict["input_ids"].shape[0]
                crl_end_pos = crl_start_pos + crl_len - 1

                data_dict["input_ids"] = torch.cat([data_dict["input_ids"], crl_tokens], dim=0)
                data_dict["labels"] = torch.cat(
                    [data_dict["labels"],
                     torch.full((crl_len,), IGNORE_INDEX, dtype=data_dict["labels"].dtype)],
                    dim=0,
                )
                ar_action_pos = data_dict["position_ids"][:, :, action_start_idx : action_end_idx + 1]
                data_dict["position_ids"] = torch.cat([data_dict["position_ids"], ar_action_pos], dim=2)

                crl_span["crl_action_start"] = crl_start_pos
                crl_span["crl_action_end_pos"] = crl_end_pos

                # --- 2. Append CRL goal tokens with <|goal_repr|> marker ---
                # Structure: Instruction: task text <|goal_repr|>
                # <|goal_repr|> captures pure instruction hidden state (self-contained attention)
                goal_tokens = self.processor(text=[f"Instruction: {item['task']}{CRL_GOAL_REPR_TOKEN}"], padding=False, return_tensors="pt")["input_ids"].squeeze(0)
                goal_len = goal_tokens.shape[0]

                crl_goal_start_pos = data_dict["input_ids"].shape[0]
                crl_goal_end_pos = crl_goal_start_pos + goal_len - 1

                data_dict["input_ids"] = torch.cat([data_dict["input_ids"], goal_tokens], dim=0)
                data_dict["labels"] = torch.cat(
                    [data_dict["labels"],
                     torch.full((goal_len,), IGNORE_INDEX, dtype=data_dict["labels"].dtype)],
                    dim=0,
                )
                # CRL goal tokens are self-contained (only attend to themselves),
                # so position_ids start from 0 like an independent text sample.
                # This ensures training-inference consistency: at inference time,
                # the instruction is encoded standalone with positions from 0.
                goal_pos_ids, _ = self.get_rope_index(
                    self.merge_size,
                    goal_tokens,
                    image_grid_thw=None,
                    video_grid_thw=None,
                    second_per_grid_ts=None,
                )
                data_dict["position_ids"] = torch.cat([data_dict["position_ids"], goal_pos_ids], dim=2)

                crl_span["crl_goal_start"] = crl_goal_start_pos
                crl_span["crl_goal_end_pos"] = crl_goal_end_pos

                # --- 3. Update seq metadata ---
                new_seq_len = data_dict["input_ids"].shape[0]
                data_dict["seq_length"] = new_seq_len
                data_dict["attention_mask"] = [new_seq_len]

                data_dict["crl_spans"] = crl_span
                data_dict["crl_weights"] = torch.tensor(self.gamma, dtype=torch.float32) ** step_to_go
                if self.use_multi_positive:
                    data_dict["crl_task_ids"] = torch.tensor(
                        instruction_to_task_id(item["task"]), dtype=torch.long
                    )

        for img in _temp_pil_images:
            img.close()
            del img
        _temp_pil_images.clear()

        return data_dict

    def info_qwen_vision_fetch(self):
        from qwen_vl_utils import smart_resize

        if not self.lerobot_dataset:
            return
        
        if self.args.rope_type == "qwen3_vl":
            image_factor = 32
        else:
            image_factor = 28

        rank0_print(f"{self.args.rope_type} min pixel {self.args.image_min_pixels}, max pixel {self.args.image_max_pixels}", msg_type='important')
        for dataset in self.lerobot_dataset._datasets:
            meta_features, video_key = dataset.meta.features, dataset.select_video_keys
            for k in video_key:
                if meta_features[k]["shape"][0] == 3:
                    h, w = meta_features[k]["shape"][1], meta_features[k]["shape"][2]
                else:
                    h, w = meta_features[k]["shape"][0], meta_features[k]["shape"][1]
                h_bar, w_bar = smart_resize(
                    h, w, factor=image_factor, min_pixels=self.args.image_min_pixels, max_pixels=self.args.image_max_pixels
                )
                rank0_print(f"{dataset.repo_id:<40} | {k:<40} | resize from {h, w} to {h_bar, w_bar} |", msg_type='info')


# TODO (zy & zjy): assign to zjy to check this part
# NOTE: only enabled during the pre-training phase
class PackedDataset(Dataset):
    """
    Performs greedy sample packing on a provided dataset. This is done as a single
    preprocessing step before training begins. Shuffling is done outside of this
    class on packed samples as part of the dataloader.

    We may randomly sample some examples from the mini action set to avoid parameter
    tracking issue(lm_head and flow head) during training.
    
    [Memory Optimized Version]:
    Avoids accumulating full sample dictionaries in a list ('items') which causes 
    double memory usage during torch.cat operations.
    """


    def __init__(
        self,
        dataset: Dataset,
        pack_length: int = 8192,
        mini_vqa_set_length: int = 512,
        pack_meta: Dict = None,
    ) -> None:
        self.dataset = dataset
        self.pack_length = pack_length
        self.mini_vqa_set_length = mini_vqa_set_length
        self.mm_dataset_len = len(dataset.mm_dataset) if hasattr(dataset, 'mm_dataset') else 0

        mode = pack_meta["mode"]
        # 兼容新旧存储格式
        if mode == "ultra":
            pack_save_path = pack_meta["grouped_indices"]
            lengths_save_path = pack_meta["lengths"]
            self.packed_indices = LeRobotStylePacker()
            self.packed_indices.load_from_parquet(pack_save_path)
            data_points_lengths = EfficientLengthStorage.load_from_disk(lengths_save_path)
        else:
            self.packed_indices = pack_meta["grouped_indices"]
            data_points_lengths = pack_meta["lengths"]

        self.mini_vqa_set = [
            idx
            for idx in range(self.mm_dataset_len)
            if data_points_lengths[idx] < self.mini_vqa_set_length
        ]       
        del data_points_lengths 

    def __len__(self):
        return len(self.packed_indices)
    
    def __getitem__(self, index: int):
        indices = self.packed_indices[index]
        
        # --- Memory Optimization Step 1: Pre-calculate indices to load ---
        # Determine if we need to append a mini vqa set sample BEFORE loading heavy data
        indices_to_load = list(indices)
        no_vqa = True
        for i in indices:
            if i < self.mm_dataset_len:
                no_vqa = False

        if no_vqa and len(self.mini_vqa_set) > 0:
            indices_to_load.append(random.choice(self.mini_vqa_set))
        # --- Memory Optimization Step 2: Initialize containers ---
        # Instead of 'items = []', we use separate lists to allow early GC of dicts
        batch_input_ids = []
        batch_labels = []
        batch_position_ids = []
        batch_attention_mask = []
        
        # Vision containers
        batch_pixel_values = []
        batch_image_grid_thw = []
        
        # Video containers
        batch_pixel_values_videos = []
        batch_video_grid_thw = []
        batch_second_per_grid_ts = []
        
        # State containers
        batch_states = []
        batch_state_dof_mask = []
        
        # CRL spans containers (with offset tracking)
        batch_crl_spans = []
        batch_crl_weights = []
        batch_crl_task_ids = []
        current_offset = 0

        # --- Memory Optimization Step 3: Streaming Load & Destructure ---
        for i in indices_to_load:
            data = self.dataset[i]
            
            seq_len = data["input_ids"].shape[0]

            # Collect basic fields
            batch_input_ids.append(data["input_ids"])
            batch_labels.append(data["labels"])
            batch_position_ids.append(data["position_ids"])
            
            if "attention_mask" in data:
                # Original logic: take the first element [0]
                batch_attention_mask.append(data["attention_mask"][0])

            # Collect Vision (Images)
            if "pixel_values" in data:
                batch_pixel_values.append(data["pixel_values"])
                batch_image_grid_thw.append(data["image_grid_thw"])
            
            # Collect Vision (Videos)
            if "pixel_values_videos" in data:
                batch_pixel_values_videos.append(data["pixel_values_videos"])
                batch_video_grid_thw.append(data["video_grid_thw"])
                if "second_per_grid_ts" in data:
                    batch_second_per_grid_ts.append(data["second_per_grid_ts"])

            # Collect States
            if "states" in data:
                batch_states.append(data["states"])
                batch_state_dof_mask.append(data["state_dof_mask"])
            
            # Collect CRL spans (adjust positions with current offset)
            if "crl_spans" in data:
                crl_span = data["crl_spans"].copy()
                # Add offset to all positions
                crl_span["instruction_start"] += current_offset
                crl_span["instruction_end"] += current_offset
                crl_span["action_start"] += current_offset
                crl_span["action_end_pos"] += current_offset
                crl_span["goal_pos"] += current_offset
                crl_span["crl_action_start"] += current_offset
                crl_span["crl_action_end_pos"] += current_offset
                if "crl_goal_start" in crl_span:
                    crl_span["crl_goal_start"] += current_offset
                    crl_span["crl_goal_end_pos"] += current_offset
                batch_crl_spans.append(crl_span)

                batch_crl_weights.append(data["crl_weights"])
                if "crl_task_ids" in data:
                    batch_crl_task_ids.append(data["crl_task_ids"])
            
            # Update offset for next sample
            current_offset += seq_len
            
            # Explicitly delete reference to allow GC to reclaim the dict wrapper
            del data

        # --- Step 4: Concatenation (Logic preserved exactly) ---
        
        # concat as a sequence
        input_ids = torch.cat(batch_input_ids, dim=0)
        labels = torch.cat(batch_labels, dim=0)
        position_ids = torch.cat(batch_position_ids, dim=2)
        
        new_data_dict = {
            "input_ids": input_ids.unsqueeze(0),
            "labels": labels.unsqueeze(0),
            "position_ids": position_ids,
            "attention_mask": batch_attention_mask if batch_attention_mask else None,
        }

        if len(batch_pixel_values) > 0:
            new_data_dict.update(
                {
                    "pixel_values": torch.cat(batch_pixel_values, dim=0),
                    "image_grid_thw": torch.cat(batch_image_grid_thw, dim=0),
                }
            )

        if len(batch_pixel_values_videos) > 0:
            new_data_dict.update(
                {
                    "pixel_values_videos": torch.cat(batch_pixel_values_videos, dim=0),
                    "video_grid_thw": torch.cat(batch_video_grid_thw, dim=0),
                    "second_per_grid_ts": torch.cat(batch_second_per_grid_ts, dim=0) if len(batch_second_per_grid_ts) > 0 else None,
                }
            )

        # NOTE: states are also required during pre-training
        if len(batch_states) > 0:
            new_data_dict.update(
                {
                    "states": torch.cat(batch_states, dim=0),
                    "state_dof_mask": torch.cat(batch_state_dof_mask, dim=0),
                }
            )
        
        # Add CRL spans if any samples have them
        if len(batch_crl_spans) > 0:
            new_data_dict["crl_spans"] = batch_crl_spans
            new_data_dict["crl_weights"] = torch.stack(batch_crl_weights, dim=0)
            if len(batch_crl_task_ids) > 0:
                new_data_dict["crl_task_ids"] = torch.stack(batch_crl_task_ids, dim=0)

        return new_data_dict

    @property
    def lerobot_dataset(self):
        return self.dataset.lerobot_dataset

    def info_qwen_vision_fetch(self):
        self.dataset.info_qwen_vision_fetch()


""" Data Collators """
class MultimodalDataCollator(DefaultDataCollator):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int, collate_position_ids: bool = True):
        self.pad_token_id = pad_token_id
        self.collate_position_ids = collate_position_ids

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []

        batch_actions = []
        batch_states = []
        batch_action_is_pad = []
        batch_state_dof_mask = []
        batch_action_dof_mask = []

        batch_dataset_names = []

        all_crl_spans = []
        all_crl_weights = []
        all_crl_task_ids = []

        batch_position_ids = []

        is_labels_provided = "labels" in examples[0]
        for batch_idx, example in enumerate(examples):
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            batch_input_ids.append(example["input_ids"])
            batch_position_ids.append(example["position_ids"].squeeze(1).permute(1, 0))
            if is_labels_provided:
                batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

            if "actions" in keys:
                batch_actions.append(example["actions"])
                batch_action_is_pad.append(example["action_is_pad"])
                batch_action_dof_mask.append(example["action_dof_mask"])

            if "states" in keys:
                batch_states.append(example["states"])
                batch_state_dof_mask.append(example["state_dof_mask"])

            if "crl_spans" in example and example["crl_spans"] is not None:
                span = example["crl_spans"].copy()
                span["batch_idx"] = batch_idx
                all_crl_spans.append(span)

            if "crl_weights" in example:
                all_crl_weights.append(example["crl_weights"])
            if "crl_task_ids" in example:
                all_crl_task_ids.append(example["crl_task_ids"])

            batch_dataset_names.append(example['dataset_names'])

        input_ids = pad_sequence(batch_input_ids, padding_side="right", padding_value=self.pad_token_id)
        attention_mask = input_ids != self.pad_token_id

        data_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "dataset_names": batch_dataset_names,
        }

        if self.collate_position_ids:
            padded = torch_pad_sequence(batch_position_ids, batch_first=False, padding_value=1)
            padded = padded.permute(2, 1, 0)
            ref = batch_position_ids[0]
            device, dtype = ref.device, ref.dtype
            position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1], dtype=dtype, device=device)
            position_ids[:, :, :padded.shape[2]] = padded
            data_dict["position_ids"] = position_ids

        if is_labels_provided:
            labels = pad_sequence(batch_label_ids, padding_side="right", padding_value=IGNORE_INDEX)
            data_dict["labels"] = labels

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

            if image_thw.ndim == 2 and image_thw.shape[0] > 0:
                image_thw_long = image_thw.to(dtype=torch.long)
                per_frame_tokens = image_thw_long[:, 1] * image_thw_long[:, 2]
                frame_tokens = per_frame_tokens.repeat_interleave(image_thw_long[:, 0])
                if frame_tokens.numel() > 0:
                    data_dict["image_max_seqlen"] = int(frame_tokens.max().item())

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        if len(batch_actions) > 0:
            actions = torch.cat(batch_actions, dim=0)
            action_is_pad = torch.cat(batch_action_is_pad, dim=0)  # (b s)
            action_dof_mask = torch.cat(batch_action_dof_mask, dim=0)

            data_dict["actions"] = actions
            data_dict["action_is_pad"] = action_is_pad
            data_dict["action_dof_mask"] = action_dof_mask

        if len(batch_states) > 0:
            states = torch.cat(batch_states, dim=0)
            state_dof_mask = torch.cat(batch_state_dof_mask, dim=0)

            data_dict["states"] = states
            data_dict["state_dof_mask"] = state_dof_mask
        
        data_dict["crl_spans"] = all_crl_spans if len(all_crl_spans) > 0 else None
        data_dict["crl_weights"] = torch.stack(all_crl_weights, dim=0) if len(all_crl_weights) > 0 else None
        data_dict["crl_task_ids"] = torch.stack(all_crl_task_ids, dim=0) if len(all_crl_task_ids) > 0 else None

        return data_dict

## qwen3-vl official
class FlattenedDataCollatorForSupervisedDataset(DefaultDataCollator):
    """Collate examples into packed sequence with multi-modal support."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:        
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        # Flatten attention_mask lists from all instances into a single list of sequence lengths
        seq_lens_list = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + seq_lens_list, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )

        if len(seq_lens_list) > 0:
            batch["max_seqlen"] = max(seq_lens_list)
        else:
            batch["max_seqlen"] = 0

        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )

        # check states
        states = list(
            instance["states"]
            for instance in instances
            if "states" in instance
        )

        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)

            if grid_thw.ndim == 2 and grid_thw.shape[0] > 0:
                grid_thw_long = grid_thw.to(dtype=torch.long)
                per_frame_tokens = grid_thw_long[:, 1] * grid_thw_long[:, 2]
                frame_tokens = per_frame_tokens.repeat_interleave(grid_thw_long[:, 0])
                if frame_tokens.numel() > 0:
                    batch["image_max_seqlen"] = int(frame_tokens.max().item())
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
            second_per_grid_ts_list = [instance["second_per_grid_ts"] for instance in instances if instance.get("second_per_grid_ts") is not None]
            if len(second_per_grid_ts_list) > 0:
                second_per_grid_ts = torch.cat(second_per_grid_ts_list, dim=0)
            else:
                second_per_grid_ts = None
        else:
            concat_videos = None
            video_grid_thw = None
            second_per_grid_ts = None

        if len(states) != 0:
            concat_states = torch.cat(states, dim=0)
            states_dof_masks = [
                instance["state_dof_mask"]
                for instance in instances
                if "state_dof_mask" in instance
            ]
            states_dof_masks = torch.cat(states_dof_masks, dim=0)
        else:
            concat_states = None
            states_dof_masks = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["second_per_grid_ts"] = second_per_grid_ts

        batch["states"] = concat_states
        batch["state_dof_mask"] = states_dof_masks
        
        # Collect CRL spans from all instances (with cross-instance offset adjustment)
        # Note: seq_lens_list contains the sequence lengths for each packed sample
        all_crl_spans = []
        all_crl_weights = []
        all_crl_task_ids = []
        cross_instance_offset = 0
        for instance in instances:
            if "crl_spans" in instance and instance["crl_spans"] is not None:
                crl_spans_data = instance["crl_spans"]
                # crl_spans can be a single dict or a list of dicts (from PackedDataset)
                if isinstance(crl_spans_data, dict):
                    crl_spans_data = [crl_spans_data]
                for span in crl_spans_data:
                    # Adjust positions with cross-instance offset
                    adjusted_span = span.copy()
                    adjusted_span["instruction_start"] += cross_instance_offset
                    adjusted_span["instruction_end"] += cross_instance_offset
                    adjusted_span["action_start"] += cross_instance_offset
                    adjusted_span["action_end_pos"] += cross_instance_offset
                    adjusted_span["goal_pos"] += cross_instance_offset
                    adjusted_span["crl_action_start"] += cross_instance_offset
                    adjusted_span["crl_action_end_pos"] += cross_instance_offset
                    if "crl_goal_start" in adjusted_span:
                        adjusted_span["crl_goal_start"] += cross_instance_offset
                        adjusted_span["crl_goal_end_pos"] += cross_instance_offset
                    all_crl_spans.append(adjusted_span)
                
                all_crl_weights.append(instance["crl_weights"])
                if "crl_task_ids" in instance:
                    all_crl_task_ids.append(instance["crl_task_ids"])
            
            # Update offset based on attention_mask (seq lengths from this instance)
            # Each instance may have multiple packed samples
            current_instance_seq_len = instance['input_ids'].size(1)
            cross_instance_offset += current_instance_seq_len
        
        batch["crl_spans"] = all_crl_spans if len(all_crl_spans) > 0 else None
        batch["crl_weights"] = torch.cat(all_crl_weights, dim=0) if len(all_crl_weights) > 0 else None
        batch["crl_task_ids"] = torch.cat(all_crl_task_ids, dim=0) if len(all_crl_task_ids) > 0 else None
        return batch
## ------------

def make_dataset(processor, args: TrainConfig, rank: int = 0):
    """build datasets and collator"""
    global local_rank
    local_rank = rank
    dataset = MultimodalLeRobotDataset(args=args, processor=processor)
    initialize_collections(dataset)

    if args.pack_dataset:
        data_collator = FlattenedDataCollatorForSupervisedDataset()
    else:
        data_collator = MultimodalDataCollator(pad_token_id=processor.tokenizer.pad_token_id)

    # Initialize the dataset collections
    return {"train_dataset": dataset, "eval_dataset": None, "data_collator": data_collator}


""" Helper Functions """
def pad_sequence(sequences, padding_side="right", padding_value=0):
    assert padding_side in ["right", "left"]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == "right":
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


### minimum requirements: qwen_vl_utils >= 0.0.14
def get_image_info(image_path, min_pixel, max_pixel, width, height, model_type = 'qwen2_5_vl'):
    content = {"type": "image", "image": image_path, "min_pixels": min_pixel, "max_pixels": max_pixel}

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    messages = [{"role": "user", "content": [content]}]

    if model_type == 'qwen2_5_vl':
        image_input, _ = process_vision_info(messages, image_patch_size=14)
    elif model_type == 'qwen3_vl':
        image_input, _ = process_vision_info(messages, image_patch_size=16)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return image_input[0]

### minimum requirements: qwen_vl_utils >= 0.0.14
def get_video_info(video_path, min_pixels, max_pixels, width, height, fps, model_type = 'qwen2_5_vl'):
    content = {
        "type": "video",
        "video": video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        # "min_frames": 30,
        # "max_frames": 60,
        "fps": fps,
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    messages = [{"role": "user", "content": [content]}]

    if model_type == 'qwen2_5_vl':
        _, video_input, video_kwargs = process_vision_info(messages, image_patch_size=14, return_video_kwargs=True)
        video_metadatas = None

    elif model_type == 'qwen3_vl':
        _, video_input, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        if video_input is not None:
            video_input, video_metadatas = zip(*video_input)
            video_input, video_metadatas = list(video_input), list(video_metadatas)
            assert len(video_input) == len(video_metadatas) == 1, f"each sample can only have one video, but got {len(video_input)} videos and {len(video_metadatas)} video metadatas"
            
            video_metadatas = video_metadatas[0]
            
        else:
            video_metadatas = None

    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return video_input[0], video_kwargs, video_metadatas


def update_processor_pixels(processor, data_args):
    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.image_min_pixels
        ip.max_pixels = data_args.image_max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.image_min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.image_max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.image_min_pixels
        ip.size["longest_edge"] = data_args.image_max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.image_min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.image_max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.fps
            rank0_print(f"✅ Updated video_processor fps to {data_args.fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor


def pack_dataset(dataset, args, seed=42, memory_mode="auto"):
    """
    使用shuffle的方式进行数据packing，支持不同的内存优化级别
    """
    pack_length = args.max_packed_length
    data_points_lengths = aggregate_dataset_length(dataset)
    data_points_lengths_list = to_list_fast(data_points_lengths)
    dataset_size = len(data_points_lengths)
    pack_meta = {}
    # 根据数据集大小和内存模式选择合适的packing函数
    if memory_mode == "auto":
        if dataset_size <= 1_000_000:  # 100万以下用标准方法
            memory_mode = "normal"
        elif dataset_size <= 10_000_000:  # 1000万以下用流式处理
            memory_mode = "streaming"
        else:  # 超过1000万用极致优化
            memory_mode = "ultra"

    pack_meta["mode"] = memory_mode
    rank0_print(f"Using memory mode: {memory_mode} for dataset size: {dataset_size:,}", msg_type='important')
    
    if memory_mode == "normal":
        pack_meta["grouped_indices"] = pack_data_points_by_length_with_shuffle(data_points_lengths_list, pack_length, seed)
        pack_meta["lengths"] = data_points_lengths
        pack_meta["appraisal"] = validate_packing_efficiency(
        packed_bins=pack_meta["grouped_indices"],
        lengths=pack_meta["lengths"],
        capacity=pack_length,
        sample_size=min(1000, dataset_size)
    )
    elif memory_mode == "streaming":
        pack_meta["grouped_indices"] = pack_data_points_by_length_with_shuffle_streaming(data_points_lengths_list, pack_length, seed)
        pack_meta["lengths"] = data_points_lengths
        pack_meta["appraisal"] = validate_packing_efficiency(
            packed_bins=pack_meta["grouped_indices"],
            lengths=pack_meta["lengths"],
            capacity=pack_length,
            sample_size=min(1000, dataset_size)
        )
    elif memory_mode == "ultra":
        grouped_indices = LeRobotStylePacker()
        pack_save_path = args.pack_indices_path if args.pack_indices_path else './configs/packing/packed_data.parquet'
        grouped_indices_list = pack_data_points_by_length_with_WFD(
            data_points_lengths_list, pack_length)
        grouped_indices.save(result=grouped_indices_list,
                             lengths=data_points_lengths,
                             max_length=pack_length,
                             output_path=pack_save_path)
        cprint(f"packed data indices saved to: {pack_save_path}", msg_type='success')
        import shutil
        data_config_path = args.data_path
        dst_config_file = os.path.join(os.path.dirname(pack_save_path), os.path.basename(data_config_path))
        shutil.copy2(data_config_path, dst_config_file)
        cprint(f"packed data config saved to: {dst_config_file}", msg_type='success')
        dst_lengths_file = os.path.join(os.path.dirname(pack_save_path), 'data_lengths.npz')
        data_points_lengths.save_to_disk(str(dst_lengths_file))
        cprint(f"data lengths saved to: {dst_lengths_file}", msg_type='success')
        pack_meta["grouped_indices"] = pack_save_path
        pack_meta["lengths"] = dst_lengths_file
        pack_meta["appraisal"] = validate_packing_efficiency(
        packed_bins=grouped_indices_list,
        lengths=data_points_lengths_list,
        capacity=pack_length,
        sample_size=min(1000, dataset_size)
        )
    else:
        raise ValueError(f"Unknown memory_mode: {memory_mode}")
    
    return pack_meta
