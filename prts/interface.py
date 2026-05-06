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
import logging
from typing import Dict, Any, Optional
from PIL import Image
from pathlib import Path
import numpy as np
from lerobot.constants import ACTION, OBS_IMAGE, OBS_STATE
from lerobot.policies.normalize import Normalize, NormalizationMode, Unnormalize

from .constants import (
    VISION_START_TOKEN,
    VISION_END_TOKEN,
    IMAGE_PLACEHOLDER_TOKEN,
    ACTION_START_TOKEN,
    ACTION_END_TOKEN,
    ACTION_PLACEHOLDER_TOKEN,
    IM_START_TOKEN,
    IM_END_TOKEN,
    SYSTEM_MESSAGE,
    IGNORE_INDEX,
)

from .data.dataset import get_image_info
from .data.normalize import load_norm_stats, load_normalizer_from_model_dir, MODEL_NORM_STATS_FILENAME
from .data.embodiment_tag import EMBODIMENT_CONFIGS
from .data.transforms import AbsoluteActionTransform

class PRTSInterface:
    """
    Policy class that combines PRTS model and processor for inference.
    Handles lerobot format observations and processes them similar to dataset.__getitem__.
    """
    
    def __init__(
        self,
        model,
        processor,
        model_path: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        image_min_pixel: int = 64 * 32 * 32,
        image_max_pixel: int = 144 * 32 * 32,
        image_resized_w: Optional[int] = None,
        image_resized_h: Optional[int] = None,
        action_dim: int = 7,
        use_fast_action: bool = False,
        state_mode: str = "MEAN_STD",
        model_type: str = "qwen3_vl",
        repo_id: Optional[str] = None,
        state_relative_action: bool = False,
        finetuned_from_crl_ckpt: bool = False,
    ):
        """
        Args:
            model_path: Path to the saved model directory.  If it contains
                ``norm_stats.json`` (written by PRTSTrainer), the normalizer is
                loaded directly from there – no dataset required.
            dataset_path: Fallback path to the original LeRobot dataset root,
                used only when ``model_path`` is not provided or does not contain
                ``norm_stats.json``.
            repo_id: When loading from ``norm_stats.json`` that covers multiple
                datasets, pick this specific repo's stats.  If ``None``, the
                first (default) entry is used.
        """
        self.model = model
        self.processor = processor
        self._last_raw_state: torch.Tensor | None = None
        self.device = device
        self.dtype = dtype
        self.image_min_pixel = image_min_pixel
        self.image_max_pixel = image_max_pixel
        self.image_resized_w = image_resized_w
        self.image_resized_h = image_resized_h
        self.action_dim = action_dim
        self.use_fast_action = use_fast_action
        self.state_mode = state_mode
        self.model_type = model_type
        self.finetuned_from_crl_ckpt = finetuned_from_crl_ckpt

        # Set model to eval mode
        self.model.eval()

        # ── Load normalizer ──────────────────────────────────────────────────
        # Priority: model_path/norm_stats.json  >  dataset_path
        loaded = False
        if model_path is not None:
            norm_stats_file = Path(model_path) / MODEL_NORM_STATS_FILENAME
            if norm_stats_file.exists():
                self.load_norm_stats_from_model_dir(model_path, repo_id=repo_id)
                loaded = True
            else:
                raise FileNotFoundError(
                    f"{MODEL_NORM_STATS_FILENAME} not found in {model_path}."
                )

        logging.info(f"PRTSInterface initialised on device: {device}")
        logging.info(f"Number of denoising steps: {self.model.config.num_denoise_steps}")

        self.state_relative_action = state_relative_action
        self.absolute_action_transform = None
        self.action_mask = None
        if self.state_relative_action:
            ecfg = EMBODIMENT_CONFIGS.get(self.model.config.embodiment_tag)
            if ecfg is not None:
                self.action_mask = torch.tensor(ecfg.delta_action_mask, dtype=torch.bool)
            else:
                raise ValueError(f"Embodiment tag {self.model.config.embodiment_tag} not found in EMBODIMENT_CONFIGS")

            self.absolute_action_transform = AbsoluteActionTransform(self.action_mask)


    def load_norm_stats_from_model_dir(
        self,
        model_path: str,
        repo_id: Optional[str] = None,
    ) -> None:
        """Load normalizer / unnormalizer from norm_stats.json in the model directory.

        This avoids instantiating a full LeRobotDataset during evaluation.

        Args:
            model_path: Directory that contains norm_stats.json.
            repo_id: Dataset repo_id to use for stats selection.  If None,
                the first (default) entry is used.
        """
        normalizer, unnormalizer = load_normalizer_from_model_dir(
            model_dir=model_path,
            state_mode=self.state_mode,
            repo_id=repo_id,
            device=self.device,
        )
        self.normalizer = normalizer
        self.unnormalizer = unnormalizer
        logging.info(f"Loaded norm stats from {model_path}/{MODEL_NORM_STATS_FILENAME}")

    def load_dataset_norm_stats(self, norm_stats_root_path: str) -> None:
        """Legacy: load normalizer by instantiating a LeRobotDataset.
        """
        from .data.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(
            'unknown',
            root=Path(norm_stats_root_path),
            state_mode=self.state_mode,
            load_quantile_stats=True,
        )
        dataset.set_unnormalization(state_mode=self.state_mode)
        self.normalizer = dataset.normalizer.to(self.device)
        self.unnormalizer = dataset.unnormalizer.to(self.device)

    
    def _process_obs(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process lerobot format observation to match the training dataset pipeline.

        States are discretized to integer text (0-1000 scale) and embedded in the
        prompt, matching the training ``__getitem__`` in ``dataset.py``.

        Also returns the raw (pre-normalisation) state tensor so that the caller
        can apply the inverse AbsoluteActionTransform when state_relative_action
        is enabled.
        """
        # Save raw state BEFORE normalisation for inverse delta transform
        raw_state_parts = []
        for k, v in obs.items():
            if k.startswith(OBS_STATE):
                raw_state_parts.append(v.clone() if isinstance(v, torch.Tensor) else torch.as_tensor(v))
        self._last_raw_state = (
            torch.cat(raw_state_parts, dim=-1).float() if raw_state_parts else None
        )

        obs = self.normalizer(obs)

        images = []
        states = []
        camera_names = []
        cam_mapping = {
            "image_0": "camera_top",
            "image_1": "camera_left_wrist",
            "image_2": "camera_right_wrist"
        }

        for k, v in obs.items():
            if k.startswith(OBS_IMAGE):
                if isinstance(v, torch.Tensor):
                    if v.dim() == 4:
                        v = v.squeeze(0)
                    image = Image.fromarray(
                        (v * 255).float().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                    )
                images.append(image)
                camera_names.append(k.split('.')[-1])

            if k.startswith(OBS_STATE):
                states.append(v)

        # Camera name mapping (same as training)
        new_camera_names = []
        for orig_name in camera_names:
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

        # Discretize state to text (matching training pipeline)
        if states:
            states_tensor = torch.cat(states, dim=-1)
            if states_tensor.dim() == 2:
                states_tensor = states_tensor.squeeze(0)
            discretized_state = ((states_tensor + 1) / 2 * 1000).to(torch.int32)
            discretized_state = torch.clamp(discretized_state, -1, 1000)
            state_str = " ".join(map(str, discretized_state.tolist()))
            state_block = f"Proprioception (normalized to 0-1000 scale): {state_str}\n"
        else:
            state_block = ""

        # current cursor give
        task_text = obs['task'][0].lower().strip().rstrip('.').capitalize() + '.'

        obs_repr_marker = "<|obs_repr|>" if self.finetuned_from_crl_ckpt else ""
        goal_repr_marker = "<|goal_repr|>" if self.finetuned_from_crl_ckpt else ""

        user_content = (
            f"{image_block}{state_block}{obs_repr_marker}"
            f"Instruction: {task_text}{goal_repr_marker}\n"
            f"Predict the next action chunk in low-level robotics action format."
        )

        conversations = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": ""},
        ]

        # Process images (pass model_type to select correct patch_size)
        processed_images = []
        for image in images:
            processed_image = get_image_info(
                image,
                self.image_min_pixel,
                self.image_max_pixel,
                self.image_resized_w,
                self.image_resized_h,
                model_type=self.model_type,
            )
            processed_images.append(processed_image)

        user_input = conversations[0]
        gpt_response = conversations[1]
        system_message = f"{IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{IM_END_TOKEN}\n"
        user_input_text = (
            f"{system_message}"
            f"{IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{IM_END_TOKEN}\n"
            f"{IM_START_TOKEN}{gpt_response['role']}\n"
        )

        inputs = self.processor(
            text=[user_input_text],
            images=processed_images,
            videos=None,
            padding=False,
            do_resize=False,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device) if "pixel_values" in inputs else None
        image_grid_thw = inputs["image_grid_thw"].to(self.device) if "image_grid_thw" in inputs else None

        attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(torch.bool)

        action_dof_mask = torch.zeros(
            1, self.model.config.action_chunk_size, self.model.config.max_action_dim,
        )
        action_dof_mask[..., :self.action_dim] = 1.0

        data_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "action_dof_mask": action_dof_mask.to(self.device),
        }
        return data_dict
    
    @torch.no_grad()
    def infer(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Infer actions from lerobot format observation.
        
        Args:
            obs: Lerobot format observation dictionary
            
        Returns:
            Dictionary containing predicted actions
        """
        # Process observation
        processed_data = self._process_obs(obs)
        # In LIBERO evaluation
        # obs dict_keys(['observation.images.image', 'observation.images.wrist_image', 'observation.state', 'task'])
        # obs['observation.images.image'] is a tensor of shape (1, 3, 224, 224), dtype = torch.bfloat16, value_range = [0, 1]
        # obs['observation.images.wrist_image'] is a tensor of shape (1, 3, 224, 224), dtype = torch.bfloat16, value_range = [0, 1]
        # obs['observation.state'] is a tensor of shape (1, 8), dtype = torch.bfloat16
        # obs['task'] is a list[str], length = 1
        
        # Run model inference
        if self.use_fast_action:
            output_ids = self.model.generate(**processed_data, max_new_tokens=1024)
        else:
            outputs, _ = self.model.sample_actions(**processed_data)
        
        # Extract action predictions
        # The exact output format depends on the model's forward method
        # This is a placeholder - you may need to adjust based on actual model output
        if self.use_fast_action:
            # For language model outputs
            action_logits = outputs.logits
            # Process action logits to get final actions
            # This depends on how your model outputs actions
            actions = self._extract_actions_from_logits(action_logits)
        else:
            # For direct action outputs
            actions = self.unnormalizer({
                "action": outputs[..., :self.action_dim].squeeze(0),
            })["action"]

        # State-relative training: unnormalized actions are still relative to current state
        # on masked dimensions — add raw proprio back (same as inverse of LeRobotDataset._apply_state_relative_transform).
        if self.state_relative_action and self.absolute_action_transform is not None:
            if self._last_raw_state is None:
                logging.warning(
                    "state_relative_action=True but no observation.state in obs; returning relative actions."
                )
            else:
                state = self._last_raw_state.to(device=actions.device, dtype=actions.dtype)
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                # AbsoluteActionTransform expects keys observation.state + action
                item = {
                    "observation.state": state,
                    "action": actions,
                }
                actions = self.absolute_action_transform(item)["action"]

        return {"actions": actions.cpu().numpy()}
    
    # def _extract_actions_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
    # TODO: Implement action extraction from model logits and policy reset functionality as needed.
