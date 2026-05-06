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

import math
import random
import collections
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.transforms import ImageTransformConfig, SharpnessJitter
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform, ColorJitter
from torchvision.transforms.v2._utils import query_size
from torchvision.transforms.v2 import functional as F  # noqa: N812
from torchvision.transforms import functional as T


class RandomScaleCrop:
    """Randomly crop the image according to the scale parameter, keeping the original aspect ratio.
    Args:
        scale: The range of size (as a fraction of the original size) to sample from.
    """

    def __init__(self, scale: tuple[float, float]):
        self.scale = scale

    def __call__(self, img: Any) -> Any:
        h, w = query_size(img)

        area = h * w
        target_area = random.uniform(self.scale[0], self.scale[1]) * area

        aspect_ratio = w / h
        crop_h = int(round(math.sqrt(target_area / aspect_ratio)))
        crop_w = int(round(target_area / crop_h))

        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)

        i = random.randint(0, h - crop_h)
        j = random.randint(0, w - crop_w)
        return v2.functional.crop(img, i, j, crop_h, crop_w)

## rhodes-lerobot-internal implementation
class DeltaActionTransform(Transform):
    """Transform absolute actions into delta action space.
    
    This class is used to convert absolute actions into delta actions by subtracting the current state
    from the action for specified dimensions (determined by action_mask).
    
    The transform assumes:
    - action shape: (batch_size, horizon, action_dim) or (horizon, action_dim) or (action_dim,)
    - state shape: (batch_size, state_dim) or (state_dim,)
    
    Args:
        action_mask: Boolean tensor indicating which action dimensions should be converted to delta.
                    1 means convert to delta, 0 means keep absolute.
    """
    def __init__(self, action_mask: torch.Tensor) -> None:
        super().__init__()
        self.action_mask = action_mask
        self.action_mask_dims = action_mask.shape[0]

    def forward(self, *inputs: Any) -> Any:
        """
        Transform the input dictionary by converting absolute actions to delta actions.
        
        Args:
            inputs: Tuple containing a single dictionary with 'observation.state' and 'action' keys
            
        Returns:
            Updated dictionary with delta actions
        """
        if len(inputs) != 1 or not isinstance(inputs[0], dict):
            return inputs

        item = inputs[0]
        if 'observation.state' not in item or 'action' not in item:
            return item

        state = item['observation.state']
        action = item['action']
        
        # 获取掩码维度
        mask = self.action_mask
        dims = self.action_mask_dims
        
        # 扩展state维度以匹配action
        # e.g., state: [batch_size, state_dim] -> [batch_size, 1, state_dim]
        if state.ndim - action.ndim == -1:
            state_expanded = state.unsqueeze(-2)
        else:
            state_expanded = state
        
        # 只对掩码维度进行delta转换
        action[..., :dims] -= torch.where(
            mask,
            state_expanded[..., :dims],
            torch.zeros_like(state_expanded[..., :dims])
        )
        
        item['action'] = action
        return item

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"action_mask={self.action_mask}"

class AbsoluteActionTransform(Transform):
    """Transform delta action space back to absolute action space.
    
    This class is used to convert delta actions back to absolute action space by adding the current state
    to the action for specified dimensions (determined by action_mask).
    
    The transform assumes:
    - action shape: (batch_size, horizon, action_dim) or (horizon, action_dim) or (action_dim,)
    - state shape: (batch_size, state_dim) or (state_dim,)
    
    Args:
        action_mask: Boolean tensor indicating which action dimensions should be converted to absolute.
                    1 means convert to absolute, 0 means keep delta.
    """
    def __init__(self, action_mask: torch.Tensor) -> None:
        super().__init__()
        self.action_mask = action_mask
        self.action_mask_dims = action_mask.shape[0]

    def forward(self, *inputs: Any) -> Any:
        """
        通过将delta动作转换为绝对动作来转换输入字典。
        
        参数:
            inputs: 包含单个字典的元组,该字典具有'observation.state'和'action'键
            
        返回:
            更新后的带有绝对动作的字典
        """
        if len(inputs) != 1 or not isinstance(inputs[0], dict):
            return inputs

        item = inputs[0]
        if 'observation.state' not in item or 'action' not in item:
            return item

        state = item['observation.state']
        action = item['action']
        
        # 获取掩码维度
        mask = self.action_mask.to(state.device)
        dims = self.action_mask_dims
        
        # 扩展state维度以匹配action
        # e.g., state: [batch_size, state_dim] -> [batch_size, 1, state_dim]
        if state.ndim - action.ndim == -1:
            state_expanded = state.unsqueeze(-2)
        else:
            state_expanded = state.clone()
        
        # 只对掩码维度进行绝对动作转换
        action[..., :dims] += torch.where(
            mask,
            state_expanded[..., :dims],
            torch.zeros_like(state_expanded[..., :dims])
        )
        
        item['action'] = action
        return item

    def extra_repr(self) -> str:
        """返回额外的表示字符串。"""
        return f"action_mask={self.action_mask}"
### ------------------------------------------------------------

# RandomSubset Transform, which is provided by the official Lerobot
class RandomSubsetApply(Transform):
    """Apply a random subset of N transformations from a list of transformations.

    Args:
        transforms: list of transformations.
        p: represents the multinomial probabilities (with no replacement) used for sampling the transform.
            If the sum of the weights is not 1, they will be normalized. If ``None`` (default), all transforms
            have the same probability.
        n_subset: number of transformations to apply. If ``None``, all transforms are applied.
            Must be in [1, len(transforms)].
        random_order: apply transformations in a random order.
    """

    def __init__(
        self,
        transforms: Sequence[Callable],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        if p is None:
            p = [1] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(
                f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}"
            )

        if n_subset is None:
            n_subset = len(transforms)
        elif not isinstance(n_subset, int):
            raise TypeError("n_subset should be an int or None")
        elif not (1 <= n_subset <= len(transforms)):
            raise ValueError(f"n_subset should be in the interval [1, {len(transforms)}]")

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]
        self.n_subset = n_subset
        self.random_order = random_order

        self.selected_transforms = None

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        self.selected_transforms = [self.transforms[i] for i in selected_indices]

        for transform in self.selected_transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"transforms={self.transforms}, "
            f"p={self.p}, "
            f"n_subset={self.n_subset}, "
            f"random_order={self.random_order}"
        )

@dataclass
class ImageTransformsConfig:
    """Image and Wrist transforms for the LeRobot dataset."""

    enable: bool = True
    max_num_transforms: int = 3
    random_order: bool = True
    tfs: dict[str, ImageTransformConfig] = field(
        default_factory=lambda: {
            "brightness": ImageTransformConfig(
                type="ColorJitter",
                kwargs={"brightness": (0.8, 1.2)},
            ),
            "contrast": ImageTransformConfig(
                type="ColorJitter",
                kwargs={"contrast": (0.8, 1.2)},
            ),
            "saturation": ImageTransformConfig(
                type="ColorJitter",
                kwargs={"saturation": (0.5, 1.5)},
            ),
            "hue": ImageTransformConfig(
                type="ColorJitter",
                kwargs={"hue": (-0.05, 0.05)},
            ),
            "sharpness": ImageTransformConfig(
                type="SharpnessJitter",
                kwargs={"sharpness": (0.5, 1.5)},
            ),
            "crop": ImageTransformConfig(
                type="RandomScaleCrop",
                kwargs={"scale": (0.95, 1.0)},
            ),
            "rotate": ImageTransformConfig(
                type="RandomRotate",
                kwargs={"degrees": (-3, 3)},
            ),
            ## LeRobot officially used to replace crop and rotate
            # "affine": ImageTransformConfig(
            #     weight=1.0,
            #     type="RandomAffine",
            #     kwargs={"degrees": (-5.0, 5.0), "translate": (0.05, 0.05)},
            # ),
        }
    )


def make_transform_from_config(cfg: ImageTransformConfig):
    if cfg.type == "Identity":
        return v2.Identity(**cfg.kwargs)
    elif cfg.type == "ColorJitter":
        return v2.ColorJitter(**cfg.kwargs)
    elif cfg.type == "SharpnessJitter":
        return SharpnessJitter(**cfg.kwargs)
    elif cfg.type == "RandomRotate":
        return v2.RandomRotation(**cfg.kwargs)
    elif cfg.type == "RandomResizedCrop":
        scale = cfg.kwargs['scale']
        size = cfg.kwargs['size']
        crop_size = [int(size[0]*scale[0]), int(size[1]*scale[1])]
        return LerobotRandomResizedCrop(crop_size=crop_size, resize_size=size)
    elif cfg.type == "RandomScaleCrop":
        return RandomScaleCrop(**cfg.kwargs)
    else:
        return v2.Identity(**cfg.kwargs)


class ImageTransforms(Transform):
    """A class to compose image transforms based on configuration."""

    def __init__(self, cfg: ImageTransformsConfig) -> None:
        super().__init__()
        self._cfg = cfg
        # self.width = width
        # self.height = height
        self.weights = []
        self.transforms = {}

        # if 'crop_resize' in cfg.tfs.keys():
        #     cfg.tfs['crop_resize'].kwargs['size'] = [height, width]

        for tf_name, tf_cfg in cfg.tfs.items():
            if tf_cfg.weight <= 0.0:
                continue
            self.transforms[tf_name] = make_transform_from_config(tf_cfg)
            self.weights.append(tf_cfg.weight)

        n_subset = min(len(self.transforms), cfg.max_num_transforms)
        if n_subset == 0 or not cfg.enable:
            self.tf = v2.Identity()
        else:
            self.tf = RandomSubsetApply(
                transforms=list(self.transforms.values()),
                p=self.weights,
                n_subset=n_subset,
                random_order=cfg.random_order,
            )

    def forward(self, *inputs: Any) -> Any:
        return self.tf(*inputs)

    @classmethod
    def create_piohfive_sequential_transform(cls, img_size: tuple[int, int]) -> v2.Compose:
        """按照固定顺序创建图像增强转换。

        Args:
            img_size: 目标图像大小 (height, width)
            p: 每个变换的概率

        Returns:
            按照 RandomCrop -> Resize -> Rotate -> ColorJitter 顺序的图像增强转换
        """
        return v2.Compose([
            v2.RandomCrop(
                size=(int(img_size[0] * 0.95), int(img_size[1] * 0.95)),
            ),
            v2.Resize(
                size=img_size,
            ),
            v2.RandomRotation(
                degrees=(-5, 5),
            ),
            LerobotColorJitter(
                brightness=(0.8, 1.2),
                contrast=(0.8, 1.2),
                saturation=(0.5, 1.5),
                hue=(-0.05, 0.05),
                p=0.5,
            )
        ])
    
    @classmethod
    def create_jax_pi0_main_camera_transform(cls) -> v2.Compose:
        """按照固定顺序创建图像增强转换。

        Args:
            img_size: 目标图像大小 (height, width)
            p: 每个变换的概率

        Returns:
            按照 RandomCrop -> Resize -> Rotate -> ColorJitter 顺序的图像增强转换
        """
        return v2.Compose([
            RandomScaleCrop(
                size=(0.95, 1.0),
            ),
            v2.RandomRotation(
                degrees=(-5, 5),
            ),
            LerobotColorJitter(
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.1,
                p=0.5,
            )
        ])

    @classmethod
    def create_jax_pi0_wrist_camera_transform(cls) -> v2.Compose:
        """按照固定顺序创建图像增强转换。

        Args:
            img_size: 目标图像大小 (height, width)
            p: 每个变换的概率

        Returns:
            按照 RandomCrop -> Resize -> Rotate -> ColorJitter 顺序的图像增强转换
        """
        return v2.Compose([
            LerobotColorJitter(
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.1,
                p=0.5,
            )
        ])
    
def dataset_to_policy_features(features: dict[str, dict]) -> dict[str, PolicyFeature]:
    policy_features = {}
    for key, ft in features.items():
        shape = ft["shape"]
        if ft["dtype"] in ["image", "video"]:
            type = FeatureType.VISUAL
            if len(shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")
            names = ft["names"]
            if names[2] in ["channel", "channels"]:  # (h, w, c) -> (c, h, w)
                shape = (shape[2], shape[0], shape[1])
        elif key == "observation.environment_state":
            type = FeatureType.ENV
        elif key.startswith("observation"):
            type = FeatureType.STATE
        elif key.startswith("action"):
            type = FeatureType.ACTION
        else:
            continue
        policy_features[key] = PolicyFeature(
            type=type,
            shape=shape,
        )
    return policy_features


## official implementation from rhodes-lerobot
class LerobotColorJitter(ColorJitter):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5):
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.p = p
    
    def forward(self, inpt):
        if self.p < 1.0:
            if torch.rand(1).item() > self.p:
                return inpt
        return super().forward(inpt)

def LerobotRandomResizedCrop(crop_size, resize_size):
    return v2.Compose([
        v2.RandomCrop(size=crop_size),
        v2.Resize(size=resize_size),
    ])