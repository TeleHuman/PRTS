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

"""Configuration classes for PRTS built on Qwen3-VL."""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig


class PRTS_Qwen3VLTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a PRTS Text Model based on Qwen3-VL.
    It extends PretrainedConfig with Qwen3-VL text model parameters and PRTS-specific parameters.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen3VL model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            Number of key-value heads for Grouped Query Attention.
        head_dim (`int`, *optional*, defaults to 128):
            The dimension of the head.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function.
        max_position_embeddings (`int`, *optional*, defaults to 128000):
            The maximum sequence length.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 5000000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        image_token_id (`int`, *optional*):
            Token index used as placeholder for image embeddings.
        video_token_id (`int`, *optional*):
            Token index used as placeholder for video embeddings.
        action_token_id (`int`, *optional*):
            Token index used as placeholder for action embeddings.
        action_start_token_id (`int`, *optional*):
            Token index for action sequence start.
        action_end_token_id (`int`, *optional*):
            Token index for action sequence end.
        vision_start_token_id (`int`, *optional*):
            Token index for vision sequence start.
        **kwargs:
            Additional keyword arguments passed to PretrainedConfig.
    """

    model_type = "prts_qwen3_vl_text"   # TODO (zy): check if this is correct
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=128000,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=5000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        # PRTS specific
        action_token_id=None,
        action_start_token_id=None,
        action_end_token_id=None,
        crl_goal_repr_token_id=None,
        crl_obs_repr_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # Validate rope config
        rope_config_validation(self, ignore_keys={"mrope_section", "mrope_interleaved"})

        # PRTS specific token IDs
        self.action_token_id = action_token_id
        self.action_start_token_id = action_start_token_id
        self.action_end_token_id = action_end_token_id
        self.crl_goal_repr_token_id = crl_goal_repr_token_id
        self.crl_obs_repr_token_id = crl_obs_repr_token_id

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class PRTS_FlowMatchingConfig_Qwen3VL(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a PRTS model based on Qwen3-VL.
    It extends PretrainedConfig with Qwen3-VL model parameters and PRTS-specific parameters for action prediction.

    [`PRTS_FlowMatchingConfig_Qwen3VL`] is the configuration class to store the configuration of a PRTS model. It is used to
    instantiate a PRTS model according to the specified arguments, defining the vision encoder, text encoder,
    action expert, and flow matching components.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.

    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `PRTS_Qwen3VLTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen3VLVisionConfig`):
            The config object or dictionary of the vision backbone.
        max_action_dim (`int`, *optional*, defaults to 14):
            Maximum dimension of action vectors. Used for padding different robot action spaces.
        action_chunk_size (`int`, *optional*, defaults to 100):
            Number of action timesteps to predict in each forward pass.
        num_denoise_steps (`int`, *optional*, defaults to 4):
            Number of denoising steps for flow matching during inference.
        flow_matching_action_loss_weight (`float`, *optional*, defaults to 1.0):
            Weight for the flow matching action loss.
        crl_loss_weight (`float`, *optional*, defaults to 0.0):
            Weight for the Contrastive Reinforcement Learning (CRL) loss. Set to 0 to disable.
        crl_embed_dim (`int`, *optional*, defaults to 256):
            Dimension of the CRL embedding space for action and goal encoders.
        crl_logsumexp_reg_weight (`float`, *optional*, defaults to 0.0):
            Weight for logsumexp regularization on CRL logits.
        image_token_id (`int`, *optional*):
            Token id for image placeholders.
        video_token_id (`int`, *optional*):
            Token id for video placeholders.
        vision_start_token_id (`int`, *optional*):
            Token id for vision start marker.
        vision_end_token_id (`int`, *optional*):
            Token id for vision end marker.
        **kwargs:
            Additional keyword arguments passed to PretrainedConfig.

    Example:

    ```python
    >>> from prts.models import PRTS_FlowMatchingConfig_Qwen3VL, PRTS_Qwen3VL

    >>> # Initializing a PRTS Qwen3-VL configuration
    >>> configuration = PRTS_FlowMatchingConfig_Qwen3VL()

    >>> # Initializing a model from the configuration
    >>> model = PRTS_Qwen3VL(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "prts_qwen3_vl"
    sub_configs = {
        "vision_config": Qwen3VLVisionConfig,
        "text_config": PRTS_Qwen3VLTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=False,
        # PRTS specific
        max_action_dim=32,
        action_chunk_size=50,
        num_denoise_steps=4,
        flow_matching_action_loss_weight=0.,
        use_fast_action_tokenizer=True,
        # Embodiment tag: identifies the robot embodiment used for finetuning.
        # Stores the delta_action_mask key so eval code can recover it without
        # needing the training dataset config.
        embodiment_tag=None,
        # DiT action head config
        dit_action_head_config=None,
        # CRL (Contrastive Reinforcement Learning) parameters
        crl_loss_weight=0.,
        crl_embed_dim=256,
        crl_logsumexp_reg_weight=0.0,
        crl_encoder_init_w=1e-12,  # Cold initialization weight for encoder last layer
        crl_repr_norm=True,  # Whether to L2-normalize CRL representations
        **kwargs,
    ):
        # Initialize vision config
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        # Initialize text config
        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # For BC use all kwargs to init `TextConfig`
            self.text_config = self.sub_configs["text_config"](**kwargs)

        # PRTS-specific parameters
        self.max_action_dim = max_action_dim
        self.action_chunk_size = action_chunk_size
        self.num_denoise_steps = num_denoise_steps
        self.flow_matching_action_loss_weight = flow_matching_action_loss_weight
        self.use_fast_action_tokenizer = use_fast_action_tokenizer
        self.embodiment_tag = embodiment_tag

        # DiT action head config (nested dict)
        # cross_attention_dim defaults to text_config.hidden_size at model init time
        _default_dit_config = {
            # Architecture — aligned with GR00T N1.6 (32 layers, inner_dim=32×48=1536)
            "num_layers": 16,   # 32
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "output_dim": 1024,
            # Regularisation
            "dropout": 0.2,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "final_dropout": True,
            # Action-head specifics
            "add_pos_embed": True,
            # Noise schedule
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            # Attention backend
            "attn_implementation": "sdpa",
            # AlternateVLDiT — separate visual / text token cross-attention
            "use_alternate_vl_dit": True,
            "attend_text_every_n_blocks": 2,
            # MoT-style action expert: forwards full VLM ``past_key_values`` into the head;
            # expert depth defaults to text_config.num_hidden_layers (override with expert_num_layers).
            "use_mot_action_expert": False,
            "mlp_mult": 4,  # FFN hidden dim = inner_dim * mlp_mult (standard DiT only)
        }
        if dit_action_head_config is not None:
            _default_dit_config.update(dit_action_head_config)
        self.dit_action_head_config = _default_dit_config

        # CRL (Contrastive Reinforcement Learning) parameters
        self.crl_loss_weight = crl_loss_weight
        self.crl_embed_dim = crl_embed_dim
        self.crl_logsumexp_reg_weight = crl_logsumexp_reg_weight
        self.crl_encoder_init_w = crl_encoder_init_w
        self.crl_repr_norm = crl_repr_norm

        # Token IDs
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        # # Propagate token IDs to text config
        # if self.image_token_id is not None:
        #     self.text_config.image_token_id = self.image_token_id
        # if self.video_token_id is not None:
        #     self.text_config.video_token_id = self.video_token_id
        # if self.vision_start_token_id is not None:
        #     self.text_config.vision_start_token_id = self.vision_start_token_id

        # Ensure vocab sizes are consistent
        # if hasattr(self.text_config, 'vocab_size'):
        #     self.vocab_size = self.text_config.vocab_size

        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)

    # TODO (zy): 这里需要看下是不是在VLConfig传入这些state action的特殊token更合适更灵活
    @property
    def action_token_id(self):
        """Get action token id from text config."""
        return getattr(self.text_config, 'action_token_id', None)

    @action_token_id.setter
    def action_token_id(self, value):
        """Set action token id in text config."""
        if hasattr(self.text_config, 'action_token_id'):
            self.text_config.action_token_id = value

    def __getattribute__(self, key):
        if "text_config" in super().__getattribute__("__dict__") and key not in [
            "dtype",
            "_attn_implementation_internal",
        ]:
            text_config = super().__getattribute__("text_config")
            if key in text_config.__dict__:
                return getattr(text_config, key)

        return super().__getattribute__(key)


PRTS_FlowMatchingConfig_Qwen3VL.register_for_auto_class()
__all__ = ["PRTS_FlowMatchingConfig_Qwen3VL", "PRTS_Qwen3VLTextConfig"]
