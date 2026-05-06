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

"""Main VLA model architecture based on Qwen3-VL."""

from dataclasses import dataclass

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers.modeling_outputs import ModelOutput
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling

from .modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
)
from .configuration_prts_qwen3_vl import PRTS_FlowMatchingConfig_Qwen3VL
from .dit_action_head import FlowMatchingDiTHead, MoTFlowMatchingHead

from prts.data.config import ACTION_DATASET_NAMES, MULTIMODAL_DATASET_NAMES
from prts.utils.logging_utils import important


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    device="cpu",
) -> torch.Tensor:
    """
    Computes sine-cosine positional embedding vectors for scalar positions (diffusion timesteps).
    
    Args:
        time: Tensor of shape (batch_size,) containing timestep values
        dimension: Embedding dimension (must be even)
        min_period: Minimum period for sinusoidal encoding
        max_period: Maximum period for sinusoidal encoding
        device: Device to create tensors on
        
    Returns:
        Positional embeddings of shape (batch_size, dimension)
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


class ContrastiveEncoder(nn.Module):
    """
    MLP projector for Contrastive Reinforcement Learning (CRL) embeddings.
    
    Projects hidden states to a shared latent space for contrastive learning,
    with L2 normalization for stable similarity computation.
    
    Architecture: N-layer MLP with LayerNorm and Swish activation, 
                  followed by a cold-initialized output projection.
                  [Linear -> LayerNorm -> Swish] x N -> Linear (cold init)

    Matches stable_contrastive_rl's Q network structure (default: 4 hidden layers).
    
    Args:
        input_dim: Dimension of input hidden states
        output_dim: Dimension of output embeddings (default: 256)
        hidden_dim: Dimension of hidden layers (default: 1024)
        num_layers: Number of hidden layers (default: 4)
        repr_norm: Whether to L2-normalize outputs (default: False)
        init_w: Small value for last layer weight initialization for cold init (default: 1e-12)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        repr_norm: bool = False,
        init_w: float = 1e-12,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.repr_norm = repr_norm
        
        # Build hidden layers with LayerNorm
        self.hidden_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.hidden_layers.append(nn.Linear(in_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Output projection layer with cold initialization
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_proj.weight.data.uniform_(-init_w, init_w)
        self.output_proj.bias.data.fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input to L2-normalized embedding space.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            L2-normalized embeddings of shape (batch_size, output_dim)
        """
        # Pass through hidden layers
        for fc, norm in zip(self.hidden_layers, self.layer_norms):
            x = fc(x)
            x = norm(x)
            x = F.silu(x)
        
        # Output projection
        x = self.output_proj(x)
        
        # Optional L2 normalization
        if self.repr_norm:
            x = F.normalize(x, dim=-1)
        
        return x



@dataclass
class PRTS_Qwen3VL_ModelOutputWithPast(ModelOutput):
    """
    Output class for PRTS model based on Qwen3-VL.
    
    Args:
        loss: Combined total loss
        flow_loss: Flow matching loss for action prediction
        cross_entropy_loss: Standard language modeling loss
        crl_loss: Contrastive Reinforcement Learning loss for goal-action alignment
        logits: Language model logits
        past_key_values: Cached key-value states
        hidden_states: Hidden states from all layers (if output_hidden_states=True)
        attentions: Attention weights (if output_attentions=True)
        rope_deltas: RoPE position delta information
        channel_loss_dict: Per-dataset loss values for logging
        channel_loss_count_dict: Per-dataset token counts for loss normalization
    """
    loss: Optional[torch.FloatTensor] = None
    flow_loss: Optional[torch.FloatTensor] = None
    cross_entropy_loss: Optional[torch.FloatTensor] = None
    crl_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None

    crl_num_samples: Optional[torch.LongTensor] = None
    channel_loss_dict: Optional[dict] = None
    channel_loss_count_dict: Optional[dict] = None


class PRTS_Qwen3VL(Qwen3VLForConditionalGeneration):
    """
    Vision-Language-Action model based on Qwen3-VL.
    
    This model extends Qwen3-VL to support:
    1. Proprioceptive state embedding and prediction
    2. Sub-task description generation (language format)
    3. Action chunk prediction via flow matching (continuous actions)
    4. Optional discrete action tokenization (fast mode)
    
    The model uses a flow matching approach for continuous action prediction, with a DiT
    (Diffusion Transformer) action head that cross-attends to VLM hidden states.
    """
    config: PRTS_FlowMatchingConfig_Qwen3VL

    _tied_weights_keys = ["lm_head.weight"]
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(
        self,
        config: PRTS_FlowMatchingConfig_Qwen3VL,
    ):
        """
        Initialize the PRTS Qwen3-VL model for action processing.

        Args:
            config: Model configuration
            use_fast_tokenizer (bool): Whether to use FAST tokenizer for discrete actions
            flow_matching_action_loss_weight (float): Weight for flow matching action loss
        """
        super().__init__(config)

        # The parent class initializes:
        # - self.visual: Qwen3VLVisionModel
        # - self.language_model: Qwen3VLTextModel
        # - self.lm_head: Language model head
        # - self.rope_deltas: Cached rope deltas
        # We keep these and add PRTS-specific components

        # PRTS-specific parameters
        self.action_dim = config.max_action_dim
        self.use_fast_tokenizer = config.use_fast_action_tokenizer
        self.flow_matching_action_loss_weight = config.flow_matching_action_loss_weight

        # Loss functions
        self.loss_fct = CrossEntropyLoss(reduction="none")
        self.loss_mse = MSELoss(reduction="none")

        # DiT-based flow matching action head: standard (+ AlternateVLDiT) or pi0.5 KV expert
        self.use_mot_action_expert = config.dit_action_head_config.get(
            "use_mot_action_expert", False
        )
        if config.flow_matching_action_loss_weight > 0.:
            if self.use_mot_action_expert:
                self.dit_action_head = MoTFlowMatchingHead(
                    action_dim=self.action_dim,
                    action_chunk_size=config.action_chunk_size,
                    vlm_config=config.text_config,
                    num_inference_timesteps=config.num_denoise_steps,
                    config=config.dit_action_head_config,
                )
            else:
                self.dit_action_head = FlowMatchingDiTHead(
                    action_dim=self.action_dim,
                    action_chunk_size=config.action_chunk_size,
                    cross_attention_dim=config.text_config.hidden_size,
                    num_inference_timesteps=config.num_denoise_steps,
                    config=config.dit_action_head_config,
                )

        # CRL (Contrastive Reinforcement Learning) components
        if config.crl_loss_weight > 0.:
            hidden_size = config.text_config.hidden_size
            # Current encoders (trainable)
            self.crl_action_encoder = ContrastiveEncoder(
                input_dim=hidden_size,
                output_dim=config.crl_embed_dim,
                init_w=config.crl_encoder_init_w,
                repr_norm=config.crl_repr_norm,
            )
            self.crl_goal_encoder = ContrastiveEncoder(
                input_dim=hidden_size,
                output_dim=config.crl_embed_dim,
                init_w=config.crl_encoder_init_w,
                repr_norm=config.crl_repr_norm,
            )
            # Learnable temperature (log-space for numerical stability, CLIP recipe).
            self.crl_logit_scale = nn.Parameter(
                torch.ones([], requires_grad=True) * math.log(1 / 0.2)
            )

        # Initialize weights
        self.post_init()

        # Print parameter counts
        visual_params = sum(p.numel() for p in self.visual.parameters())
        language_params = sum(p.numel() for p in self.language_model.parameters())
        model_params = visual_params + language_params
        important(f"Backbone VLM (visual + language_model) parameters: {model_params / 1e6:.2f}M")
        important(f"Flow Matching Loss coefficient: {self.flow_matching_action_loss_weight}")

        if config.flow_matching_action_loss_weight > 0.:
            dit_params = sum(p.numel() for p in self.dit_action_head.parameters())
            # Get the inner model type name for logging
            if hasattr(self.dit_action_head, 'dit'):
                dit_head_type = type(self.dit_action_head.dit).__name__
            else:
                dit_head_type = type(self.dit_action_head).__name__
            important(f"DiT Action Head ({dit_head_type}) parameters: {dit_params / 1e6:.2f}M")

        if config.crl_loss_weight > 0.:
            crl_params = sum(p.numel() for p in self.crl_action_encoder.parameters())
            crl_params += sum(p.numel() for p in self.crl_goal_encoder.parameters())
            important(f"CRL Encoders (action + goal) parameters: {crl_params / 1e6:.2f}M")
            important(f"CRL Loss coefficient: {config.crl_loss_weight}")
            important(f"CRL Encoder init_w: {config.crl_encoder_init_w}")
            important(f"CRL Repr Norm: {config.crl_repr_norm}")

        self.fast_action_token_start_idx = 200000
        self.use_multi_positive = True

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)
    
    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def to_float32_flow_matching_head(self):
        """Convert flow matching heads to float32 for numerical stability."""
        if hasattr(self, 'dit_action_head'):
            self.dit_action_head = self.dit_action_head.to(dtype=torch.float32)

    def set_fast_action_info(self, action_mapper, fast_action_token_start_idx):
        """Set information for fast (discrete) action tokenization."""
        self.action_mapper = action_mapper
        self.fast_action_token_start_idx = fast_action_token_start_idx
    
    def get_placeholder_mask_with_special_token(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        special_features: torch.FloatTensor,
        special_pad_token_id: int,
    ):
        """
        Get placeholder mask for a specific special token (e.g., state tokens).
        
        Similar to get_placeholder_mask but for custom special tokens beyond image/video.
        """
        if input_ids is None:
            special_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(special_pad_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_mask = special_mask.all(-1)
        else:
            special_mask = input_ids == special_pad_token_id

        n_special_tokens = special_mask.sum()
        special_mask = special_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if special_features is not None and inputs_embeds[special_mask].numel() != special_features.numel():
            raise ValueError(
                f"Features and tokens do not match: tokens: {n_special_tokens}, features {special_features.shape[0]}"
            )

        return special_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        # use_cache: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        # rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        actions: Optional[torch.Tensor] = None,
        action_is_pad: torch.Tensor | None = None,
        action_dof_mask: Optional[torch.Tensor] = None,
        dataset_names: Optional[List[str]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, PRTS_Qwen3VL_ModelOutputWithPast]:
        """
        Forward pass for PRTS_Qwen3VL model.
        
        This extends Qwen3VLForConditionalGeneration.forward with:
        - State embedding injection
        - Action chunk flow matching
        - DeepStack visual feature handling
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")


        # 1. Prepare input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        # 2. Process images with deepstack features
        deepstack_image_embeds = None
        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw, image_max_seqlen=kwargs['image_max_seqlen'])
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # 3. Process videos with deepstack features
        deepstack_video_embeds = None
        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # 4. Aggregate deepstack visual features
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        # 7. Calculate position IDs using Qwen3VL's rope index
        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        _lm_extra_kwargs: dict = {}

        _use_cache = (
            self.use_mot_action_expert
            and self.flow_matching_action_loss_weight > 0.
            and actions is not None
        )

        vlm_outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=_use_cache,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            output_hidden_states=False,
            **_lm_extra_kwargs,
            **kwargs,
        )

        vlm_hidden_states = vlm_outputs.last_hidden_state

        # 11. Run DiT action head if actions are present
        dit_pred_v = None
        dit_velocity = None
        if actions is not None and self.flow_matching_action_loss_weight > 0:
            # vlm_hidden_states shape: bs, seq_length, hidden_size
            actions_for_dit = actions.to(vlm_hidden_states.device, dtype=vlm_hidden_states.dtype)
            dof_mask_for_dit = action_dof_mask.to(vlm_hidden_states.device, dtype=vlm_hidden_states.dtype) if action_dof_mask is not None else None
            # Pass attention_mask so DiT cross-attention ignores padding tokens
            dit_encoder_attention_mask = attention_mask.bool() if attention_mask is not None else None

            if self.use_mot_action_expert and vlm_outputs.past_key_values is not None:
                dit_pred_v, dit_velocity = self.dit_action_head(
                    vlm_outputs.past_key_values,
                    actions_for_dit,
                    dof_mask_for_dit,
                    encoder_attention_mask=dit_encoder_attention_mask,
                )
            else:
                # Standard: pass single (last-layer) VLM hidden states
                dit_image_mask = visual_pos_masks.bool() if visual_pos_masks is not None else None
                dit_pred_v, dit_velocity = self.dit_action_head(
                    vlm_hidden_states, actions_for_dit, dof_mask_for_dit,
                    encoder_attention_mask=dit_encoder_attention_mask,
                    image_mask=dit_image_mask,
                )

        # 12. Compute logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(vlm_hidden_states[:, slice_indices, :])
        
        # 13. Compute losses
        loss = None
        cross_entropy_loss, flow_loss = None, None
        channel_loss_dict = None
        channel_loss_count_dict = None

        if labels is not None:
            loss = 0
            action_accuracy = 0
            unique_datasets_name = list(set(dataset_names)) if dataset_names is not None else []

            # Compute cross-entropy loss
            shift_logits = logits[..., :-1, :].float().contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            non_ignored_mask = shift_labels != -100
            _cross_entropy_loss = self.loss_fct(shift_logits, shift_labels)
            cross_entropy_loss = (
                _cross_entropy_loss[non_ignored_mask].mean()
                if non_ignored_mask.any()
                else (_cross_entropy_loss.sum() * 0.0)
            )

            # Add cross-entropy loss to total
            if not torch.isnan(cross_entropy_loss):
                loss += cross_entropy_loss
            else:
                with torch.no_grad():
                    cross_entropy_loss.detach()

            # Compute action token prediction accuracy (for logging)
            shift_logits_for_acc = logits[..., :-1, :].contiguous()
            action_preds = shift_logits_for_acc.argmax(dim=-1)
            shift_labels_for_acc = labels[..., 1:].contiguous()

            action_mask = (
                shift_labels_for_acc >= self.fast_action_token_start_idx
            )

            if self.use_fast_tokenizer and action_mask.any():
                correct_preds = (action_preds == shift_labels_for_acc) & action_mask
                action_accuracy = (
                    correct_preds.sum().float() / action_mask.sum().float()
                )

                if channel_loss_dict is None:
                    channel_loss_dict = {}
                    channel_loss_count_dict = {}
                
                channel_loss_dict["action_accuracy"] = action_accuracy.detach()
                channel_loss_count_dict["action_accuracy"] = torch.tensor(1, device=action_accuracy.device)

        # 14. Compute flow matching loss (DiT action head)
        if dit_pred_v is not None and self.flow_matching_action_loss_weight > 0:
            if channel_loss_dict is not None:
                channel_loss_dict.update(
                    {
                        f"flow_matching/{dataset_name}": torch.tensor(0.0, device=logits.device)
                        for dataset_name in ACTION_DATASET_NAMES
                    }
                )
                channel_loss_count_dict.update(
                    {
                        f"flow_matching/{dataset_name}": torch.tensor(0, device=logits.device)
                        for dataset_name in ACTION_DATASET_NAMES
                    }
                )
            else:
                channel_loss_dict = {
                    f"flow_matching/{dataset_name}": torch.tensor(0.0, device=logits.device)
                    for dataset_name in ACTION_DATASET_NAMES
                }
                channel_loss_count_dict = {
                    f"flow_matching/{dataset_name}": torch.tensor(0, device=logits.device)
                    for dataset_name in ACTION_DATASET_NAMES
                }

            # Compute flow matching loss: MSE between predicted and target velocity
            _fm_loss = self.loss_mse(dit_pred_v, dit_velocity)

            # Apply DOF mask (zero out invalid action dimensions)
            if action_dof_mask is not None:
                valid_action_dim = int(action_dof_mask[0, 0, :].sum(dim=-1).item())     # NOTE: only support 单种具身实体数据微调
                _fm_loss = _fm_loss[:, :, :valid_action_dim]

            # Apply action_is_pad mask: exclude padding timesteps from loss
            # action_is_pad: (B, T), True = pad timestep → should not contribute to loss
            if action_is_pad is not None:
                valid_timestep_mask = ~action_is_pad[:, :_fm_loss.shape[1]]  # align length
                _fm_loss = _fm_loss * valid_timestep_mask.unsqueeze(-1)
                flow_loss = _fm_loss.sum() / (valid_timestep_mask.sum() * _fm_loss.shape[-1])
            else:
                flow_loss = _fm_loss.mean()
            
            if not torch.isnan(flow_loss):
                loss = loss + self.flow_matching_action_loss_weight * flow_loss if loss is not None else self.flow_matching_action_loss_weight * flow_loss
            else:
                with torch.no_grad():
                    flow_loss.detach()

            # Per-dataset flow matching loss logging
            logging_fm_loss = _fm_loss.detach().mean(dim=(1, 2))  # Sum over chunk_size and action_dim

            action_dataset_names = dataset_names if dataset_names is not None else []
            unique_action_datasets = list(set(action_dataset_names))
            
            for dataset_name_i in unique_action_datasets:
                action_dataset_mask = torch.tensor(
                    [name == dataset_name_i for name in action_dataset_names],
                    device=logits.device,
                )
                if action_dataset_mask.any():
                    dataset_fm_loss = logging_fm_loss[action_dataset_mask].sum()
                    dataset_fm_count = action_dataset_mask.sum()
                    
                    prefixed_key = f"flow_matching/{dataset_name_i}"
                    channel_loss_dict[prefixed_key] += dataset_fm_loss
                    channel_loss_count_dict[prefixed_key] += dataset_fm_count
            
        elif self.flow_matching_action_loss_weight > 0: 
            # Dummy loss to keep all DiT parameters in computation graph
            dummy_params = [p.sum() * 0.0 for p in self.dit_action_head.parameters() if p.requires_grad]
            dummy_loss = sum(dummy_params) if len(dummy_params) > 0 else torch.tensor(0.0, device=logits.device)
            loss = (loss + dummy_loss) if loss is not None else dummy_loss

        return PRTS_Qwen3VL_ModelOutputWithPast(
            loss=loss,
            cross_entropy_loss=(
                cross_entropy_loss.detach() if cross_entropy_loss is not None else None
            ),
            flow_loss=(
                flow_loss.detach() if flow_loss is not None else None
            ),
            crl_loss=None,
            logits=logits,
            past_key_values=vlm_outputs.past_key_values,
            # hidden_states=vlm_outputs.hidden_states,
            # attentions=vlm_outputs.attentions,
            crl_num_samples=None,
            rope_deltas=self.rope_deltas,
            channel_loss_dict=channel_loss_dict,
            channel_loss_count_dict=channel_loss_count_dict,
        )
    

    def embed_prefix(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Embed prefix tokens including vision, DeepStack, and (optionally) state features.

        Returns:
            (inputs_embeds, visual_pos_masks, deepstack_visual_embeds)
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(
                pixel_values, image_grid_thw,
                image_max_seqlen=kwargs.get('image_max_seqlen'),
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        return inputs_embeds, visual_pos_masks, deepstack_visual_embeds

    @torch.no_grad()
    def sample_actions(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        action_dof_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Sample actions using DiT-based flow matching denoising.

        1. Computes position_ids via get_rope_index
        2. Embeds the prefix (with DeepStack visual features)
        3. Runs the language model to get hidden states
        4. Uses DiT action head to denoise actions via cross-attention to VLM features

        Returns:
            (x_t, outputs) — denoised action trajectories and language-model outputs
        """
        if position_ids is None:
            position_ids, _ = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if inputs_embeds is None:
            inputs_embeds, visual_pos_masks, deepstack_visual_embeds = self.embed_prefix(
                input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                **kwargs,
            )

        _sample_use_cache = (
            self.use_mot_action_expert and self.flow_matching_action_loss_weight > 0
        )
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=_sample_use_cache,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            output_hidden_states=False,
        )

        vlm_hidden_states = outputs.last_hidden_state
        dit_encoder_attention_mask = attention_mask.bool() if attention_mask is not None else None

        if self.use_mot_action_expert and outputs.past_key_values is not None:
            x_t = self.dit_action_head.predict_action(
                outputs.past_key_values,
                action_dof_mask,
                encoder_attention_mask=dit_encoder_attention_mask,
            )
        else:
            dit_image_mask = visual_pos_masks.bool() if visual_pos_masks is not None else None
            x_t = self.dit_action_head.predict_action(
                vlm_hidden_states, action_dof_mask,
                encoder_attention_mask=dit_encoder_attention_mask,
                image_mask=dit_image_mask,
            )

        return x_t, outputs

    def frozen_crl_modules(self):
        for p in self.crl_goal_encoder.parameters():
            p.requires_grad_(False)
        for p in self.crl_action_encoder.parameters():
            p.requires_grad_(False)

        self.crl_logit_scale.requires_grad_(False)


PRTS_Qwen3VL.register_for_auto_class()


__all__ = ["PRTS_Qwen3VL", "PRTS_Qwen3VL_ModelOutputWithPast"]