from typing import List, Dict, Optional
from dataclasses import dataclass, field
# from qwen_vl_utils.vision_process import MIN_PIXELS, MAX_PIXELS, IMAGE_FACTOR

MULTIMODAL_DATASET_NAMES = []
ACTION_DATASET_NAMES = []

def initialize_collections(dataset):
    if len(dataset.lerobot_dataset) > 0:
        ACTION_DATASET_NAMES.extend(dataset.lerobot_dataset.repo_ids)
    
    if len(dataset.mm_dataset) > 0:
        MULTIMODAL_DATASET_NAMES.extend(dataset.mm_dataset.collection_names)
