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

"""This module defines constants used throughout the application,
including system messages and various special tokens for language
and vision models.
These tokens are used to demarcate different types of input such
as images, videos, actions, and states, with specific sets for
different model architectures like LLaVA and datasets like LeRobot.
"""

PRETRAINING_PHASE = "pre-training"
POSTTRAINING_PHASE = "post-training"


SYSTEM_MESSAGE = "You are a helpful physical assistant."

# qwen2.5-vl special tokens
IM_START_TOKEN = "<|im_start|>"     # beginning of turn
IM_END_TOKEN = "<|im_end|>"         # end of turn
IMAGE_PLACEHOLDER_TOKEN = "<|image_pad|>"       # image placeholder
VIDEO_PLACEHOLDER_TOKEN = "<|video_pad|>"       # video placeholder
VISION_START_TOKEN = "<|vision_start|>"     # beginning of vision input
VISION_END_TOKEN = "<|vision_end|>"         # end of vision input

# “<|endoftext|>” is inserted after each document to signify that the document \
# has ended and a new document will proceed.
PAD_TOKEN = "<|endoftext|>"

# PRTS special tokens
## For discrete action token
ACTION_START_TOKEN = "<|action_start|>"
ACTION_PLACEHOLDER_TOKEN = "<|action_pad|>"
ACTION_END_TOKEN = "<|action_end|>"
CRL_GOAL_REPR_TOKEN = "<|goal_repr|>"
CRL_OBS_REPR_TOKEN = "<|obs_repr|>"

# llava style special tokens
IGNORE_INDEX = -100
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
LLAVA_ACTION_TOKEN = "<action>"