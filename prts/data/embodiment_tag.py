"""Embodiment tag registry for managing per-robot action representations.

Each EmbodimentConfig specifies:
  - delta_action_mask: per-dim boolean list indicating which action dimensions use
    state-relative representation (True) vs absolute (False).
  - state_relative: if True (default), masked dims use state-relative representation
    (action - current_state). If False, uses sequential delta (action[t] - action[t-1]).

The delta_action_mask will be truncated to the actual action dimensionality at runtime,
so you can specify a mask longer than needed.

Usage in training YAML:
    lerobot_datasets:
      - repo_id: ...
        embodiment_tag: agibot_dual_arm
        state_relative_action: true
"""

from dataclasses import dataclass


@dataclass
class EmbodimentConfig:
    """Per-embodiment configuration for action representation.

    Attributes:
        name: Human-readable name.
        delta_action_mask: Per-dim boolean list. True = subtract current state
            (or previous action for sequential-delta mode). False = keep absolute.
            Will be truncated to the actual concatenated action dim at runtime.
        state_relative: If True, masked dims use ``action[t] - state[t]`` (state-
            relative). If False, uses sequential delta ``action[t] - action[t-1]``.
            Note: sequential-delta mode is handled by ``LeRobotDataset.set_delta_action``
            and this flag is informational only for that case.
    """

    name: str
    delta_action_mask: list[bool]
    state_relative: bool = True


# ---------------------------------------------------------------------------
# Pre-registered embodiment configs. Add new robots here.
# The mask is truncated to the actual action dim, so it's safe to have a mask
# longer than the actual number of action dimensions.
# ---------------------------------------------------------------------------
EMBODIMENT_CONFIGS: dict[str, EmbodimentConfig] = {
    "libero_panda": EmbodimentConfig(
        name="libero_panda",
        delta_action_mask=[True] * 6 + [False] * 1,
        state_relative=True,
    ),
    "flexiv": EmbodimentConfig(
        name="flexiv",
        delta_action_mask=[True] * 7 + [False] * 1,
        state_relative=True,
    ),
    "realman_dual_arm": EmbodimentConfig(
        name="realman_dual_arm",
        delta_action_mask=[True] * 14 + [False] * 2,
        state_relative=True,
    ),
    "arx_dual_arm": EmbodimentConfig(
        name="arx_dual_arm",
        delta_action_mask=[True] * 6 + [False] * 1 + [True] * 6 + [False] * 1,
        state_relative=True,
    ),
    "galaxea_r1_pro": EmbodimentConfig(
        name="galaxea_r1_pro",
        delta_action_mask=[True] * 7 + [False] * 1 + [True] * 7 + [False] * 1,
        state_relative=True,
    ),
    "agibot_g2": EmbodimentConfig(
        name="agibot_g2",
        delta_action_mask=[True] * 14 + [False] * 7,
        state_relative=True,
    ),
    # ── Generic ──────────────────────────────────────────────────────────────
    # All dims state-relative (mask truncated at runtime to actual action_dim)
    "full_state_relative": EmbodimentConfig(
        name="full_state_relative",
        delta_action_mask=[True] * 32,
        state_relative=True,
    ),
    # All dims absolute (no transformation)
    "full_absolute": EmbodimentConfig(
        name="full_absolute",
        delta_action_mask=[False] * 32,
        state_relative=False,
    ),
}
