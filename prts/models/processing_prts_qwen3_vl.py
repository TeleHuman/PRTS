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

"""Processor for PRTS built on Qwen3-VL."""

import logging
from typing import Optional, Union
from colorlog import ColoredFormatter

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import (
    ImagesKwargs,
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
    VideosKwargs,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils.logging import get_logger
from transformers.video_utils import VideoInput

from ..constants import (
    ACTION_START_TOKEN,
    ACTION_PLACEHOLDER_TOKEN,
    ACTION_END_TOKEN,
    CRL_GOAL_REPR_TOKEN,
    CRL_OBS_REPR_TOKEN,
    ## qwen3-vl tokens
    VISION_START_TOKEN,
    IMAGE_PLACEHOLDER_TOKEN,
    VIDEO_PLACEHOLDER_TOKEN,
)
from ..training.train_utils import LOG_FORMATTER

logger = get_logger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(LOG_FORMATTER)
    logger.addHandler(handler)


class Qwen3VLVideosProcessorKwargs(VideosKwargs, total=False):
    pass


class Qwen3VLImagesKwargs(ImagesKwargs):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]


class Qwen3VLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Qwen3VLImagesKwargs
    videos_kwargs: Qwen3VLVideosProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
    }


class PRTS_Qwen3VLProcessor(ProcessorMixin):
    r"""
    Constructs a PRTS processor which wraps a Qwen3-VL image processor and a Qwen2 tokenizer into a single processor.
    
    This processor is built independently (not inheriting from Qwen3VLProcessor) to avoid tight coupling,
    while maintaining compatibility with Qwen3-VL's timestamp-based video processing approach.
    
    [`PRTS_Qwen3VLProcessor`] offers all the functionalities needed for PRTS model with:
    - Action token handling (discrete and continuous)
    - State token handling for proprioceptive inputs
    - Expert trigger tokens for flow matching action prediction
    - Qwen3-VL compatible image/video processing with timestamp-based video handling
    
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        video_processor ([`Qwen3VLVideoProcessor`], *optional*):
            The video processor is a required input.
        chat_template (`str`, *optional*):
            A Jinja template which will be used to convert lists of messages in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer", "video_processor"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None,
                 chat_template=None, **kwargs):
        # Initialize base ProcessorMixin
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)
        
        # Get image/video tokens from tokenizer
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        
        # Qwen3-VL vision tokens
        self.vision_start_token = (
            "<|vision_start|>" if not hasattr(tokenizer, "vision_start_token") else tokenizer.vision_start_token
        )
        self.vision_end_token = (
            "<|vision_end|>" if not hasattr(tokenizer, "vision_end_token") else tokenizer.vision_end_token
        )
        self.vision_start_token_id = (
            tokenizer.vision_start_token_id
            if getattr(tokenizer, "vision_start_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_start_token)
        )
        self.vision_end_token_id = (
            tokenizer.vision_end_token_id
            if getattr(tokenizer, "vision_end_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_end_token)
        )

        # Add PRTS-specific special tokens
        prts_special_tokens = [
            ACTION_START_TOKEN,
            ACTION_PLACEHOLDER_TOKEN,
            ACTION_END_TOKEN,
            CRL_GOAL_REPR_TOKEN,
            CRL_OBS_REPR_TOKEN,
        ]
        num_new_tokens = tokenizer.add_tokens(prts_special_tokens, special_tokens=True)
        logger.info(f"Added {num_new_tokens} new special tokens to the tokenizer.")

        # Store PRTS-specific token information
        self.action_token = getattr(tokenizer, "action_token", ACTION_PLACEHOLDER_TOKEN)
        self.action_token_id = tokenizer.convert_tokens_to_ids(self.action_token)
        # Create a dictionary of all token IDs
        token_dict = {
            "action_start_token_id": ACTION_START_TOKEN,
            "action_token_id": ACTION_PLACEHOLDER_TOKEN,
            "vision_start_token_id": VISION_START_TOKEN,
            "image_token_id": IMAGE_PLACEHOLDER_TOKEN,
            "video_token_id": VIDEO_PLACEHOLDER_TOKEN,
            # crl special
            "crl_goal_repr_token_id": CRL_GOAL_REPR_TOKEN,
            "crl_obs_repr_token_id": CRL_OBS_REPR_TOKEN,
        }
        self.token_ids = {key: tokenizer.convert_tokens_to_ids(value) for key, value in token_dict.items()}

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: Optional[VideoInput] = None,
        actions: Union[torch.Tensor] = None,
        **kwargs: Unpack[Qwen3VLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s)/video(s). This method forwards
        the `text` and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not
        `None` to encode the text. To prepare the vision inputs, this method forwards the `images`/`videos` and
        `kwargs` arguments to the respective processors.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The video or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            actions (`torch.Tensor`, *optional*):
                Action tensor for training (not processed by this method, passed through).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            Qwen3VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_grid_thw = None

        videos_inputs = {}
        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
            # Get video metadata for timestamp calculation
            if "return_metadata" not in kwargs:
                video_metadata = videos_inputs.pop("video_metadata", None)
            else:
                video_metadata = videos_inputs.get("video_metadata", None)
        else:
            video_grid_thw = None
            video_metadata = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place
        
        # Process image tokens
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        # Process video tokens with Qwen3-VL timestamp-based approach
        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    if video_metadata is not None and index < len(video_metadata):
                        metadata = video_metadata[index]
                        if metadata.fps is None:
                            logger.warning_once(
                                "Qwen3VL requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                                "Probably `video_metadata` was missing from inputs and you passed pre-sampled frames. "
                                "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
                            )
                            metadata.fps = 24 if metadata.fps is None else metadata.fps

                        # Calculate timestamps for this video
                        curr_timestamp = self._calculate_timestamps(
                            metadata.frames_indices,
                            metadata.fps,
                            self.video_processor.merge_size,
                        )

                        video_placeholder = ""
                        frame_seqlen = video_grid_thw[index][1:].prod() // merge_length
                        for frame_idx in range(video_grid_thw[index][0]):
                            curr_time = curr_timestamp[frame_idx]
                            video_placeholder += f"<{curr_time:.1f} seconds>"
                            video_placeholder += (
                                self.vision_start_token + "<|placeholder|>" * frame_seqlen + self.vision_end_token
                            )
                        
                        # Replace video token with timestamped version
                        if f"{self.vision_start_token}{self.video_token}{self.vision_end_token}" in text[i]:
                            text[i] = text[i].replace(
                                f"{self.vision_start_token}{self.video_token}{self.vision_end_token}", video_placeholder, 1
                            )
                        else:
                            # vllm may input video token directly
                            text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                    else:
                        # Fallback: no metadata, use simple replacement
                        num_video_tokens = video_grid_thw[index].prod() // merge_length
                        text[i] = text[i].replace(self.video_token, "<|placeholder|>" * num_video_tokens, 1)
                    
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        # Add actions to output if provided (pass through, not processed here)
        output_data = {**text_inputs, **image_inputs, **videos_inputs}
        if actions is not None:
            output_data["actions"] = actions

        return BatchFeature(data=output_data, tensor_type=return_tensors)

    def _calculate_timestamps(self, indices: Union[list[int], np.ndarray], video_fps: float, merge_size: int = 2):
        """
        Calculate timestamps for video frames based on frame indices and fps.
        
        This is used for Qwen3-VL's timestamp-based video processing approach.
        
        Args:
            indices: Frame indices in the video
            video_fps: Frames per second of the video
            merge_size: Temporal patch merge size
            
        Returns:
            List of timestamps for each temporal patch
        """
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        timestamps = [idx / video_fps for idx in indices]
        # Frames are merged by merge_size, so we need to average the timestamps
        # between the first/last frame within the temporal patch
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.
        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.
            video_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (num_frames, height, width) per each video.
        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """

        vision_data = {}
        if image_sizes is not None:
            images_kwargs = Qwen3VLProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [(num_patches // merge_size**2) for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        if video_sizes is not None:
            videos_kwargs = Qwen3VLProcessorKwargs._defaults.get("videos_kwargs", {})
            videos_kwargs.update(kwargs)
            merge_size = videos_kwargs.get("merge_size", None) or self.video_processor.merge_size
            num_video_patches = [
                self.video_processor.get_number_of_video_patches(*video_size, videos_kwargs)
                for video_size in video_sizes
            ]
            num_video_tokens = [(num_patches // merge_size**2) for num_patches in num_video_patches]
            vision_data["num_video_tokens"] = num_video_tokens

        return MultiModalData(**vision_data)

    def set_action_tokenizer(self, action_tokenizer):
        """
        Set the action tokenizer for fast action token generation.
        
        This method adds discrete action tokens to the tokenizer vocabulary, allowing the model
        to predict actions as discrete tokens instead of continuous values.
        
        Args:
            action_tokenizer: The action tokenizer instance (e.g., UniversalActionProcessor)
        """
        self.action_tokenizer = action_tokenizer
        
        # Add fast action tokens to vocabulary
        prts_fast_action_tokens = [f"<|action_token_{i}|>" for i in range(action_tokenizer.vocab_size)]
        num_new_tokens = self.tokenizer.add_tokens(prts_fast_action_tokens, special_tokens=True)
        logger.info(f"Added {num_new_tokens} FAST action tokens to the tokenizer.")
        
        # Store action token information
        self.action_token_start_index = self.tokenizer.convert_tokens_to_ids("<|action_token_0|>")
        self.action_vocab_size = action_tokenizer.vocab_size

        # Create a mapping from action token strings to IDs
        token_ids = self.tokenizer.convert_tokens_to_ids(prts_fast_action_tokens)
        self.action_mapper = {k: v for k, v in zip(prts_fast_action_tokens, token_ids, strict=True)}

    def preprocess_action(self, actions, **kwargs):
        """
        Preprocess action data for model input.
        
        This method should be implemented based on your specific action preprocessing needs.
        
        Args:
            actions: Raw action data
            **kwargs: Additional preprocessing arguments
            
        Raises:
            NotImplementedError: This method needs to be implemented based on your use case.
        """
        raise NotImplementedError

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape
                `(batch_size, sequence_length)` or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's
                `batch_decode` method.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the tokenization spaces. Argument passed to the tokenizer's
                `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode` method.

        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self):
        """Get the list of model input names."""
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        names_from_processor = list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
        return names_from_processor


__all__ = ["PRTS_Qwen3VLProcessor"]
