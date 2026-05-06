"""
DiT (Diffusion Transformer) based flow matching action head for PRTS.

Replaces the Qwen3VLTextModel-based fm_action_expert with a lightweight DiT
that uses explicit cross-attention to VLM hidden states, following the architecture
from GR00T / pi05.

Architecture:
    ActionEncoder(noisy_actions + dof_mask, timestep)
    → action_features
    → DiT(cross-attn to VLM hidden states, ada-norm timestep conditioning)
    → ActionDecoder → predicted velocity
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from typing import Optional

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward


# DIT_PRESETS = {
#     "DiT-B": {"num_attention_heads": 12, "attention_head_dim": 64, "output_dim": 768},
#     "DiT-L": {"num_attention_heads": 32, "attention_head_dim": 48, "output_dim": 1536},
# }


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions or timesteps."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        squeeze = False
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(1)
            squeeze = True

        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * (
            math.log(10000.0) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        enc = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)

        if squeeze:
            enc = enc.squeeze(1)
        return enc


class TimestepEncoder(nn.Module):
    """Projects scalar timesteps to embedding space via sinusoidal encoding + MLP."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalPositionalEncoding(256)
        self.linear_1 = nn.Linear(256, embedding_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = self.sinusoidal(timesteps)
        t_emb = self.linear_1(t_emb.to(dtype=self.linear_1.weight.dtype))
        t_emb = self.act(t_emb)
        t_emb = self.linear_2(t_emb)
        return t_emb


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on timestep embeddings.

    Applies scale-shift modulation: out = norm(x) * (1 + scale) + shift,
    where (scale, shift) are linearly projected from the timestep embedding.
    """

    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=False)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class DiTAttention(nn.Module):
    """Multi-head attention supporting both self-attention and cross-attention.

    Supports two backends selected via ``attn_implementation``:

    * ``"sdpa"`` (default) – uses :func:`F.scaled_dot_product_attention`, which
      dispatches automatically to FlashAttention / memory-efficient attention
      depending on the installed PyTorch build.  The encoder padding mask is
      expanded to ``(B, 1, 1, S)`` and passed as ``attn_mask``.

    * ``"flash_attention_2"`` – calls the ``flash_attn`` package directly for
      lower memory usage and higher throughput.  For cross-attention with an
      encoder padding mask the k/v tensors are unpadded and
      :func:`flash_attn_varlen_func` is used so that padding tokens are never
      processed.  For self-attention (no mask) the simpler
      :func:`flash_attn_func` is used.
    """

    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        head_dim: int,
        cross_attention_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        attn_implementation: str = "sdpa",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_implementation = attn_implementation
        inner_dim = num_heads * head_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        kv_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.to_k = nn.Linear(kv_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(kv_dim, inner_dim, bias=bias)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=bias),
            nn.Dropout(dropout),
        )

    # ------------------------------------------------------------------
    # Flash-Attention backend
    # ------------------------------------------------------------------

    def _flash_attn_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run Flash Attention via HuggingFace's ``_flash_attention_forward``.

        Args:
            q: ``(B, T_q, H, D)``
            k: ``(B, T_k, H, D)``
            v: ``(B, T_k, H, D)``
            attention_mask: ``(B, T_k)`` bool, True = valid token.

        Returns:
            ``(B, T_q, H*D)``
        """

        B, T_q, H, D = q.shape
        # _flash_attention_forward returns (B, T_q, H, D); handles unpad/varlen internally.
        out = _flash_attention_forward(
            q, k, v,
            attention_mask=attention_mask,
            query_length=T_q,
            is_causal=False,
            dropout=0.0,
        )
        return out.reshape(B, T_q, H * D)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        q = self.to_q(hidden_states)
        kv_input = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        if self.attn_implementation == "flash_attention_2":
            # Flash Attention expects (B, S, H, D)
            q = q.view(B, T, self.num_heads, self.head_dim)
            k = k.view(B, -1, self.num_heads, self.head_dim)
            v = v.view(B, -1, self.num_heads, self.head_dim)
            attn_output = self._flash_attn_forward(q, k, v, attention_mask)
        else:
            # SDPA expects (B, H, S, D)
            q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

            # Expand (B, S) bool mask → (B, 1, 1, S) for broadcasting.
            sdpa_mask = None
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    sdpa_mask = attention_mask[:, None, None, :]
                else:
                    sdpa_mask = attention_mask

            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=sdpa_mask, dropout_p=0.0
            )
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)

        return self.to_out(attn_output)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, dim: int, dropout: float = 0.0, mult: int = 4):
        super().__init__()
        inner_dim = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    """Transformer block with self/cross-attention, optional AdaLayerNorm, and feed-forward.

    When cross_attention_dim is set, the attention block performs cross-attention
    to encoder_hidden_states. Otherwise, it performs self-attention.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        norm_type: str = "ada_norm",
        final_dropout: bool = False,
        attn_implementation: str = "sdpa",
    ):
        super().__init__()
        self.norm_type = norm_type

        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim)

        self.attn1 = DiTAttention(
            query_dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            dropout=dropout,
            attn_implementation=attn_implementation,
        )

        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)
        self.final_dropout = nn.Dropout(dropout) if final_dropout else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )

        if self.final_dropout is not None:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states


class DiT(nn.Module):
    """Diffusion Transformer with cross-attention to VLM context features.

    Interleaves cross-attention blocks (attending to encoder_hidden_states)
    with self-attention blocks when interleave_self_attention=True.
    Uses AdaLayerNorm for timestep conditioning throughout.

    Output block applies timestep-conditioned scale-shift before final projection.
    """

    def __init__(
        self,
        num_attention_heads: int = 12,
        attention_head_dim: int = 64,
        output_dim: int = 768,
        num_layers: int = 12,
        dropout: float = 0.1,
        norm_type: str = "ada_norm",
        final_dropout: bool = True,
        interleave_self_attention: bool = False,
        cross_attention_dim: Optional[int] = None,
        attn_implementation: str = "sdpa",
    ):
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.interleave_self_attention = interleave_self_attention

        self.timestep_encoder = TimestepEncoder(self.inner_dim)

        all_blocks = []
        for idx in range(num_layers):
            use_self_attn = idx % 2 == 1 and interleave_self_attention
            curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None

            all_blocks.append(
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                    norm_type=norm_type,
                    final_dropout=final_dropout,
                    attn_implementation=attn_implementation,
                )
            )
        self.transformer_blocks = nn.ModuleList(all_blocks)

        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.timestep_encoder(timestep)

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    temb=temb,
                )

        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=-1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)


class AlternateVLDiT(DiT):
    """DiT variant that separates visual and text tokens during cross-attention.

    Mirrors GR00T's AlternateVLDiT: even-indexed blocks do cross-attention,
    alternating every ``attend_text_every_n_blocks`` between text tokens and
    visual tokens.  Odd-indexed blocks do self-attention (requires
    ``interleave_self_attention=True``).

    When no visual tokens are present (``image_mask`` is None or all-False),
    all valid tokens are treated as text.
    """

    def __init__(self, *args, attend_text_every_n_blocks: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.interleave_self_attention, (
            "AlternateVLDiT requires interleave_self_attention=True"
        )
        self.attend_text_every_n_blocks = attend_text_every_n_blocks

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_attention_mask: (B, S) bool – True = valid VLM token.
            image_mask: (B, S) bool – True = visual token position.
                If None, all valid tokens are treated as text.
        """
        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        B, S, _ = encoder_hidden_states.shape
        backbone_mask = (
            encoder_attention_mask.bool()
            if encoder_attention_mask is not None
            else torch.ones(B, S, dtype=torch.bool, device=hidden_states.device)
        )

        if image_mask is not None and image_mask.any():
            vis_mask  = image_mask.bool() & backbone_mask   # visual tokens
            text_mask = (~image_mask.bool()) & backbone_mask # text tokens
        else:
            # No visual tokens – treat everything as text.
            vis_mask  = torch.zeros_like(backbone_mask)
            text_mask = backbone_mask

        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                # Self-attention block.
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                # Cross-attention block: alternate text / visual every N blocks.
                if idx % (2 * self.attend_text_every_n_blocks) == 0:
                    curr_mask = text_mask
                else:
                    curr_mask = vis_mask
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=curr_mask,
                    temb=temb,
                )

        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=-1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)


class ActionEncoder(nn.Module):
    """Encodes noisy actions (optionally concatenated with DOF mask) and timestep
    into hidden features via MLP + sinusoidal time encoding.

    Architecture: Linear → concat(action_emb, time_emb) → SiLU + Linear → Linear
    """

    def __init__(self, action_input_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(action_input_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: (B, T, action_input_dim) noisy actions (+ DOF mask)
            timesteps: (B,) discretized timesteps
        """
        B, T, _ = actions.shape
        timesteps_expanded = timesteps.unsqueeze(1).expand(-1, T)

        a_emb = self.layer1(actions)
        tau_emb = self.pos_encoding(timesteps_expanded).to(dtype=a_emb.dtype)

        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = F.silu(self.layer2(x))
        x = self.layer3(x)
        return x


class ActionDecoder(nn.Module):
    """2-layer MLP that decodes DiT output to action-space velocity."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(x)))


class FlowMatchingDiTHead(nn.Module):
    """Flow matching action head using DiT (Diffusion Transformer).

    Replaces the fm_action_expert (Qwen3VLTextModel-based) with a DiT that uses
    explicit cross-attention to VLM hidden states instead of KV cache continuation.

    Training:
        1. Sample noise and timestep from Beta distribution
        2. Compute noisy trajectory: x_t = (1-t)*noise + t*actions
        3. Compute velocity target: v = actions - noise
        4. Encode noisy actions + DOF mask + timestep → action features
        5. Prepend learned future query tokens
        6. Run DiT with cross-attention to VLM hidden states
        7. Decode to action-space velocity prediction

    Inference:
        Euler integration from pure noise (t=0) to clean actions (t=1)
        over num_inference_timesteps steps.
    """

    def __init__(
        self,
        action_dim: int,
        action_chunk_size: int,
        cross_attention_dim: int,
        num_inference_timesteps: int = 4,
        config: Optional[dict] = None,
    ):
        super().__init__()
        cfg = {
            "num_layers": 16,
            "num_attention_heads": 12,
            "attention_head_dim": 64,
            "output_dim": 1024,
            "dropout": 0.2,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "final_dropout": True,
            "add_pos_embed": True,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "attn_implementation": "sdpa",
            "use_alternate_vl_dit": False,
            "attend_text_every_n_blocks": 2,
        }
        if config is not None:
            cfg.update(config)
            # dit_model_type = config.get("dit_model_type")
            # if dit_model_type and dit_model_type in DIT_PRESETS:
            #     cfg.update(DIT_PRESETS[dit_model_type])
        # cfg.pop("dit_model_type", None)

        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.num_inference_timesteps = num_inference_timesteps
        self.num_timestep_buckets = cfg["num_timestep_buckets"]
        self.noise_s = cfg["noise_s"]
        self.use_alternate_vl_dit = cfg["use_alternate_vl_dit"]
        self.add_pos_embed = cfg["add_pos_embed"]

        num_attention_heads = cfg["num_attention_heads"]
        attention_head_dim = cfg["attention_head_dim"]
        output_dim = cfg["output_dim"]
        inner_dim = num_attention_heads * attention_head_dim

        dit_kwargs = dict(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            output_dim=output_dim,
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            norm_type=cfg["norm_type"],
            final_dropout=cfg["final_dropout"],
            interleave_self_attention=cfg["interleave_self_attention"],
            cross_attention_dim=cross_attention_dim,
            attn_implementation=cfg["attn_implementation"],
        )
        if self.use_alternate_vl_dit:
            self.dit = AlternateVLDiT(
                **dit_kwargs,
                attend_text_every_n_blocks=cfg["attend_text_every_n_blocks"],
            )
        else:
            self.dit = DiT(**dit_kwargs)

        # action_dim * 2: noisy action + DOF mask concatenated
        self.action_encoder = ActionEncoder(action_dim * 2, inner_dim)
        self.action_decoder = ActionDecoder(output_dim, inner_dim, action_dim)

        if self.add_pos_embed:
            max_seq_len = max(action_chunk_size, 256)
            self.position_embedding = nn.Embedding(max_seq_len, inner_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # self.beta_dist = Beta(cfg["noise_beta_alpha"], cfg["noise_beta_beta"])
        self._beta_alpha = cfg["noise_beta_alpha"]
        self._beta_beta  = cfg["noise_beta_beta"]

    def reset_parameters(self):
        """Re-apply proper initialization.

        HuggingFace from_pretrained calls _init_weights on modules whose
        parameters are absent from the checkpoint, overwriting any custom
        init done in __init__.  Call this after from_pretrained when loading
        from a base VLM checkpoint that does not contain DiT weights.
        """
        if self.add_pos_embed:
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)
            elif isinstance(module, nn.LayerNorm):
                if module.elementwise_affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def sample_time(self, batch_size: int, device, dtype) -> torch.Tensor:
        beta_dist = Beta(self._beta_alpha, self._beta_beta)
        sample = beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.noise_s)
        return (self.noise_s - sample) / self.noise_s

    def _encode_actions(
        self,
        noisy_actions: torch.Tensor,
        t_discretized: torch.Tensor,
        action_dof_mask: Optional[torch.Tensor],
        device,
    ) -> torch.Tensor:
        """Encode noisy actions with DOF mask and timestep, add position embeddings."""
        if action_dof_mask is not None:
            encoder_input = torch.cat(
                [noisy_actions, action_dof_mask.to(noisy_actions.dtype)], dim=-1
            )
        else:
            encoder_input = torch.cat(
                [noisy_actions, torch.ones_like(noisy_actions)], dim=-1
            )

        action_features = self.action_encoder(encoder_input, t_discretized)

        if self.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        return action_features

    def _dit_forward(
        self,
        sa_embs: torch.Tensor,
        vl_embs: torch.Tensor,
        t_discretized: torch.LongTensor,
        encoder_attention_mask: Optional[torch.Tensor],
        image_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.use_alternate_vl_dit:
            return self.dit(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=t_discretized,
                encoder_attention_mask=encoder_attention_mask,
                image_mask=image_mask,
            )
        return self.dit(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=t_discretized,
            encoder_attention_mask=encoder_attention_mask,
        )

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        action_dof_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Training forward pass.

        Args:
            vl_embs: (B, S, D) VLM hidden states for cross-attention
            actions: (B, T, action_dim) ground truth action trajectories
            action_dof_mask: (B, T, action_dim) DOF validity mask
            encoder_attention_mask: (B, S) bool – True = valid VLM token
            image_mask: (B, S) bool – True = visual token (used by AlternateVLDiT)

        Returns:
            (pred_v, velocity): predicted velocity and target velocity, both (B, T, action_dim)
        """
        device = vl_embs.device
        B = actions.shape[0]

        noise = torch.randn(actions.shape, device=device, dtype=actions.dtype)
        t = self.sample_time(B, device=device, dtype=actions.dtype)
        t_expanded = t[:, None, None]

        noisy_trajectory = (1 - t_expanded) * noise + t_expanded * actions
        velocity = actions - noise

        t_discretized = (t * self.num_timestep_buckets).long()

        action_features = self._encode_actions(noisy_trajectory, t_discretized, action_dof_mask, device)

        model_output = self._dit_forward(
            action_features, vl_embs, t_discretized, encoder_attention_mask, image_mask
        )

        pred = self.action_decoder(model_output)
        pred_v = pred[:, :actions.shape[1]]

        return pred_v, velocity

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        action_dof_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference: denoise actions from noise using Euler integration.

        Args:
            vl_embs: (B, S, D) VLM hidden states
            action_dof_mask: optional (B, T, action_dim) or (1, T, action_dim) DOF mask
            encoder_attention_mask: (B, S) bool – True = valid VLM token
            image_mask: (B, S) bool – True = visual token (used by AlternateVLDiT)

        Returns:
            (B, T, action_dim) denoised action trajectories
        """
        B = vl_embs.shape[0]
        device = vl_embs.device
        dtype = vl_embs.dtype

        actions = torch.randn(
            (B, self.action_chunk_size, self.action_dim),
            device=device, dtype=dtype,
        )

        dt = 1.0 / self.num_inference_timesteps

        for step in range(self.num_inference_timesteps):
            t_cont = step / float(self.num_inference_timesteps)
            t_discretized_val = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full((B,), t_discretized_val, device=device, dtype=torch.long)

            action_features = self._encode_actions(actions, timesteps_tensor, action_dof_mask, device)

            model_output = self._dit_forward(
                action_features, vl_embs, timesteps_tensor, encoder_attention_mask, image_mask
            )

            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, :self.action_chunk_size]

            actions = actions + dt * pred_velocity

        return actions


# ============================================================================
# Pi0.5-style KV-cache action expert (VLM K/V concat + GQA + SwiGLU FFN)
# ============================================================================
class AdaRMSNorm(nn.Module):
    """Adaptive RMS normalization: (scale, shift, gate) from cond; zero-init."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.modulation = nn.Linear(dim, dim * 3)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        var = x.float().pow(2).mean(-1, keepdim=True)
        normed = (x * torch.rsqrt(var + self.eps)).to(x.dtype)
        scale, shift, gate = self.modulation(cond).chunk(3, dim=-1)
        normed = normed * (1 + scale[:, None]) + shift[:, None]
        return normed, gate[:, None]


class SwiGLUFeedForward(nn.Module):
    """SiLU(gate_proj(x)) * up_proj(x) → down_proj."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class MoTAttention(nn.Module):
    """Action Q attends to concatenated [VLM KV cache ; action KV]; GQA expand for SDPA."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        if num_attention_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        q_dim = num_attention_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, q_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=bias)
        self.o_proj = nn.Linear(q_dim, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        action_hidden: torch.Tensor,
        vlm_cached_k: torch.Tensor,
        vlm_cached_v: torch.Tensor,
        vlm_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T_a, _ = action_hidden.shape

        q = self.q_proj(action_hidden)
        act_k = self.k_proj(action_hidden)
        act_v = self.v_proj(action_hidden)

        q = q.view(B, T_a, self.num_attention_heads, self.head_dim).transpose(1, 2)
        act_k = act_k.view(B, T_a, self.num_kv_heads, self.head_dim).transpose(1, 2)
        act_v = act_v.view(B, T_a, self.num_kv_heads, self.head_dim).transpose(1, 2)

        k = torch.cat([vlm_cached_k, act_k], dim=2)
        v = torch.cat([vlm_cached_v, act_v], dim=2)

        repeat_factor = self.num_attention_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

        sdpa_mask = None
        if vlm_attention_mask is not None:
            action_mask = vlm_attention_mask.new_ones(B, T_a)
            combined_mask = torch.cat([vlm_attention_mask, action_mask], dim=1)
            sdpa_mask = combined_mask[:, None, None, :]

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=sdpa_mask, dropout_p=0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_a, -1)
        return self.dropout(self.o_proj(attn_out))


class MoTBlock(nn.Module):
    """AdaRMSNorm → attention → gated residual → AdaRMSNorm → SwiGLU FFN → gated residual."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pre_attn_norm = AdaRMSNorm(hidden_size)
        self.attn = MoTAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.pre_ffn_norm = AdaRMSNorm(hidden_size)
        self.ffn = SwiGLUFeedForward(hidden_size, intermediate_size, dropout=dropout)

    def forward(
        self,
        action_hidden: torch.Tensor,
        vlm_cached_k: torch.Tensor,
        vlm_cached_v: torch.Tensor,
        adarms_cond: torch.Tensor,
        vlm_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed, gate1 = self.pre_attn_norm(action_hidden, adarms_cond)
        attn_out = self.attn(normed, vlm_cached_k, vlm_cached_v, vlm_attention_mask)
        action_hidden = action_hidden + attn_out * gate1

        normed2, gate2 = self.pre_ffn_norm(action_hidden, adarms_cond)
        action_hidden = action_hidden + self.ffn(normed2) * gate2
        return action_hidden


class MoTDiT(nn.Module):
    """Stack of ActionBlocks; each block uses one VLM layer's KV pair."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        num_layers: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.blocks = nn.ModuleList([
            MoTBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = AdaRMSNorm(hidden_size)

    def forward(
        self,
        action_hidden: torch.Tensor,
        vlm_kv_cache: list,
        adarms_cond: torch.Tensor,
        vlm_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for idx, block in enumerate(self.blocks):
            cached_k, cached_v = vlm_kv_cache[idx]
            action_hidden = block(
                action_hidden, cached_k, cached_v, adarms_cond, vlm_attention_mask,
            )
        action_hidden, _ = self.final_norm(action_hidden, adarms_cond)
        return action_hidden


def _kv_pairs_from_past_key_values(past_key_values: Cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per-layer (K, V) from a HuggingFace decoder KV cache (order matches transformer layers)."""
    return [
        (past_key_values[i][0], past_key_values[i][1])
        for i in range(len(past_key_values))
    ]


class MoTFlowMatchingHead(nn.Module):
    """Flow matching head: MoT-style action expert over VLM KV cache (concat + GQA)."""

    def __init__(
        self,
        action_dim: int,
        action_chunk_size: int,
        vlm_config,
        num_inference_timesteps: int = 10,
        config: Optional[dict] = None,
    ):
        super().__init__()

        _vlm_num_q_heads  = 8 # vlm_config.num_attention_heads // 2 # optional: 8
        _vlm_num_kv_heads = vlm_config.num_key_value_heads   # 8
        _vlm_head_dim     = getattr(
            vlm_config, "head_dim", vlm_config.hidden_size // vlm_config.num_attention_heads
        )  # 128

        cfg = {
            "hidden_size": 1024, # vlm_config.hidden_size // 2,
            # "hidden_size": vlm_config.hidden_size // 2,
            "intermediate_size": vlm_config.intermediate_size // 4,
            "expert_num_layers": vlm_config.num_hidden_layers,
            # Attention dims default to VLM values (required for KV cache compat)
            "num_attention_heads": _vlm_num_q_heads,
            "num_kv_heads": _vlm_num_kv_heads,
            "head_dim": _vlm_head_dim,
            # Noise schedule
            "dropout": 0.2,
            "add_pos_embed": True,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
        }
        if config is not None:
            config = cfg.copy()

        num_attention_heads = cfg["num_attention_heads"]
        num_kv_heads        = cfg["num_kv_heads"]
        head_dim            = cfg["head_dim"]
        hidden_size         = cfg["hidden_size"]
        intermediate_size   = cfg["intermediate_size"]
        num_layers          = cfg["expert_num_layers"]

        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.num_inference_timesteps = num_inference_timesteps
        self.num_timestep_buckets = cfg["num_timestep_buckets"]
        self.noise_s = cfg["noise_s"]
        self.add_pos_embed = cfg["add_pos_embed"]

        self.action_in_proj = nn.Linear(action_dim * 2, hidden_size)
        self.action_out_proj = nn.Linear(hidden_size, action_dim)

        self.time_sinusoidal = SinusoidalPositionalEncoding(hidden_size)
        self.time_mlp_1 = nn.Linear(hidden_size, hidden_size)
        self.time_mlp_2 = nn.Linear(hidden_size, hidden_size)

        if self.add_pos_embed:
            max_seq = max(action_chunk_size, 256)
            self.position_embedding = nn.Embedding(max_seq, hidden_size)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.dit = MoTDiT(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
            dropout=cfg["dropout"],
        )

        self._beta_alpha = cfg["noise_beta_alpha"]
        self._beta_beta = cfg["noise_beta_beta"]

    @property
    def num_dit_layers(self) -> int:
        """Number of expert blocks; must match ``len(past_key_values.key_cache)``."""
        return self.dit.num_layers

    def _vlm_kv_list_from_past(self, past_key_values: Cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
        n = len(past_key_values)
        if n != self.num_dit_layers:
            raise ValueError(
                f"MoT expert has {self.num_dit_layers} blocks but `past_key_values` has {n} "
                "layers. Set `dit_action_head_config['expert_num_layers']` to match "
                "`text_config.num_hidden_layers`."
            )
        return _kv_pairs_from_past_key_values(past_key_values)

    def reset_parameters(self):
        """Re-apply proper initialization after from_pretrained."""
        if self.add_pos_embed:
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, AdaRMSNorm):
                nn.init.zeros_(module.modulation.weight)
                nn.init.zeros_(module.modulation.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)

    def _compute_adarms_cond(self, t_discretized: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_sinusoidal(t_discretized.float())
        t_emb = t_emb.to(dtype=self.time_mlp_1.weight.dtype)
        t_emb = F.silu(self.time_mlp_1(t_emb))
        t_emb = F.silu(self.time_mlp_2(t_emb))
        return t_emb

    def sample_time(self, batch_size: int, device, dtype) -> torch.Tensor:
        beta_dist = Beta(self._beta_alpha, self._beta_beta)
        sample = beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.noise_s)
        return (self.noise_s - sample) / self.noise_s

    def _prepare_action_embeds(
        self,
        noisy_actions: torch.Tensor,
        action_dof_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if action_dof_mask is not None:
            x = torch.cat(
                [noisy_actions, action_dof_mask.to(noisy_actions.dtype)], dim=-1,
            )
        else:
            x = torch.cat([noisy_actions, torch.ones_like(noisy_actions)], dim=-1)

        tokens = self.action_in_proj(x)

        if self.add_pos_embed:
            pos_ids = torch.arange(tokens.shape[1], dtype=torch.long, device=noisy_actions.device)
            tokens = tokens + self.position_embedding(pos_ids).unsqueeze(0)

        return tokens

    def forward(
        self,
        past_key_values: Cache,
        actions: torch.Tensor,
        action_dof_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Training: returns (pred_velocity, target_velocity).

        Args:
            past_key_values: VLM decoder KV cache; layer count must equal ``num_dit_layers``.
        """
        vlm_kv_cache = self._vlm_kv_list_from_past(past_key_values)
        device = actions.device
        B = actions.shape[0]

        noise = torch.randn(actions.shape, device=device, dtype=actions.dtype)
        t = self.sample_time(B, device=device, dtype=actions.dtype)
        t_expanded = t[:, None, None]

        noisy_trajectory = (1 - t_expanded) * noise + t_expanded * actions
        velocity = actions - noise

        t_discretized = (t * self.num_timestep_buckets).long()
        adarms_cond = self._compute_adarms_cond(t_discretized)

        action_tokens = self._prepare_action_embeds(noisy_trajectory, action_dof_mask)

        output = self.dit(
            action_tokens, vlm_kv_cache, adarms_cond, encoder_attention_mask,
        )

        pred = self.action_out_proj(output)
        pred_v = pred[:, :actions.shape[1]]
        return pred_v, velocity
    
    def compute_velocity(
        self,
        past_key_values: Cache,
        actions: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        action_dof_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute velocity prediction for pre-sampled noise and timestep.

        Args:
            past_key_values: VLM decoder KV cache
            actions: (B, T, action_dim) ground truth actions (x_0)
            noise: (B, T, action_dim) pre-sampled noise (ε)
            t: (B,) continuous timesteps in [0, 1)
            action_dof_mask, encoder_attention_mask,

        Returns:
            pred_v: (B, T, action_dim) predicted velocity
        """
        vlm_kv_cache = self._vlm_kv_list_from_past(past_key_values)
        device = actions.device
        t_expanded = t[:, None, None]

        noisy_trajectory = (1 - t_expanded) * noise + t_expanded * actions
        t_discretized = (t * self.num_timestep_buckets).long()
        adarms_cond = self._compute_adarms_cond(t_discretized)
        action_tokens = self._prepare_action_embeds(noisy_trajectory, action_dof_mask)
        output = self.dit(
            action_tokens, vlm_kv_cache, adarms_cond, encoder_attention_mask,
        )
        pred = self.action_out_proj(output)
        return pred[:, :actions.shape[1]]


    @torch.no_grad()
    def predict_action(
        self,
        past_key_values: Cache,
        action_dof_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference: Euler integration, returns (B, chunk_size, action_dim)."""
        k0 = past_key_values[0][0]
        B = k0.shape[0]
        device = k0.device
        dtype = k0.dtype
        vlm_kv_cache = self._vlm_kv_list_from_past(past_key_values)

        actions = torch.randn(
            (B, self.action_chunk_size, self.action_dim),
            device=device, dtype=dtype,
        )
        dt = 1.0 / self.num_inference_timesteps

        for step in range(self.num_inference_timesteps):
            t_cont = step / float(self.num_inference_timesteps)
            t_disc_val = int(t_cont * self.num_timestep_buckets)
            t_tensor = torch.full((B,), t_disc_val, device=device, dtype=torch.long)

            adarms_cond = self._compute_adarms_cond(t_tensor)
            action_tokens = self._prepare_action_embeds(actions, action_dof_mask)

            output = self.dit(
                action_tokens, vlm_kv_cache, adarms_cond, encoder_attention_mask,
            )
            pred_velocity = self.action_out_proj(output)[:, :self.action_chunk_size]
            actions = actions + dt * pred_velocity

        return actions