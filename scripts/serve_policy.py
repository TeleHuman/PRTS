import dataclasses
import enum
import logging
import socket

import tyro

from openpi_client import websocket_policy_server
from accelerate.state import PartialState

from prts.models.modeling_prts_qwen3_vl import PRTS_Qwen3VL
from prts.models.processing_prts_qwen3_vl import PRTS_Qwen3VLProcessor
from prts.models.processing_action_tokenizer import UniversalActionProcessor
from prts.models.configuration_prts_qwen3_vl import PRTS_FlowMatchingConfig_Qwen3VL
from prts.interface import PRTSInterface

from torchvision.transforms import v2
import torch
import os

# Calculate and print model size
def calculate_model_size(model):
    """计算模型的参数量和大小"""
    total_params = 0
    trainable_params = 0
    total_size = 0
    frozen_param_names = []
    
    for name, param in model.named_parameters():
        param_size = param.numel()
        total_params += param_size
        if param.requires_grad:
            trainable_params += param_size
        else:
            frozen_param_names.append(name)
        
        if param.dtype == torch.float32:
            bytes_per_param = 4
        elif param.dtype == torch.float16 or param.dtype == torch.bfloat16:
            bytes_per_param = 2
        elif param.dtype == torch.int8:
            bytes_per_param = 1
        elif param.dtype == torch.int32:
            bytes_per_param = 4
        elif param.dtype == torch.int64:
            bytes_per_param = 8
        else:
            bytes_per_param = 4
        
        total_size += param_size * bytes_per_param
    
    return total_params, trainable_params, total_size, frozen_param_names


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    SIMPLER = "simplerenv"

@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str

    action_dim: int

    dataset_path: str

    state_mode: str = "MEAN_STD"

    state_relative_action: bool = False


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 10000  # 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi0_aloha",
        dir="gs://openpi-assets/checkpoints/pi0_base",
        action_dim=7,
        dataset_path=None,
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
        action_dim=7,
        dataset_path=None,
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="gs://openpi-assets/checkpoints/pi0_fast_droid",
        action_dim=7,
        dataset_path=None,
    ),
    EnvMode.LIBERO: Checkpoint(
        config="prts_libero",
        dir="/path/to/your_libero_sft_ckpt",
        action_dim=7,
        dataset_path='/path/to/libero_4_suites',
        state_mode="QUANTILE",
    ),
}

def create_default_policy(env: EnvMode, *, default_prompt: str | None = None):
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        print(f"load from: {checkpoint.dir}")

        # load prts
        ## load processor
        processor = PRTS_Qwen3VLProcessor.from_pretrained(
            checkpoint.dir,
            padding_side="right",
            use_fast=True,
        )
        action_tokenizer = UniversalActionProcessor.from_pretrained(
            f"{os.environ['HF_HUB_CACHE']}/models--physical-intelligence--fast", 
            trust_remote_code=True
        )
        processor.set_action_tokenizer(action_tokenizer)

        ## load config
        config = PRTS_FlowMatchingConfig_Qwen3VL.from_pretrained(
            checkpoint.dir,
        )

        model = PRTS_Qwen3VL.from_pretrained(
            checkpoint.dir,
            config=config,
            dtype="auto",
            device_map="auto",
        )
        model.set_fast_action_info(processor.action_mapper, processor.action_token_start_index)

        total_params, trainable_params, total_size_bytes, frozen_param_names = calculate_model_size(model)
        total_size_mb = total_size_bytes / (1024 * 1024)
        total_size_gb = total_size_bytes / (1024 * 1024 * 1024)

        print("=" * 60)
        print("模型统计信息:")
        print(f"总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
        print(f"可训练参数量: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
        print(f"冻结参数量: {total_params - trainable_params:,} ({(total_params - trainable_params) / 1e6:.2f}M)")
        print(f"模型大小: {total_size_mb:.2f} MB ({total_size_gb:.2f} GB)")
        print("=" * 60)

        if frozen_param_names:
            print(f"\n冻结参数列表 (共 {len(frozen_param_names)} 个):")
            print("-" * 60)
            for name in frozen_param_names:
                print(f"  - {name}")
            print("-" * 60)
        else:
            print("\n没有冻结的参数")
        
        # Set model to evaluation mode for inference
        model.eval()
        print("\n✓ Model set to evaluation mode")

        # Create PRTS policy
        policy = PRTSInterface(
            model=model,
            processor=processor,
            model_path=checkpoint.dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16,
            action_dim=checkpoint.action_dim,
            #
            image_min_pixel=processor.image_processor.min_pixels,
            image_max_pixel=processor.image_processor.max_pixels,
            state_mode=checkpoint.state_mode,
            state_relative_action=checkpoint.state_relative_action,
        )
        
        return policy

    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args):
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            raise ValueError(f"Not Implemented yet")
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    # only for fix accelerate logger errors
    PartialState()

    policy = create_policy(args)
    
    # Set up policy metadata and transforms for PRTS policy
    if isinstance(policy, PRTSInterface):
        policy_metadata = {}
        action_mask = torch.ones(7, dtype=torch.bool)
        action_mask[-1] = False
        # Note: You may need to import AbsoluteActionTransform if available
        # action_transform = AbsoluteActionTransform(action_mask)
        action_transform = None  # Placeholder - adjust based on your needs
    else:
        raise ValueError(f"Unsupported policy: {type(policy)}")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
        delta2abs=action_transform,
    )
    server.serve_forever()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))