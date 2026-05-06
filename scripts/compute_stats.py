from collections.abc import Callable
from pathlib import Path

from lerobot.constants import ACTION, HF_LEROBOT_HOME, OBS_STATE
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset as BaseLeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset as BaseMultiLeRobotDataset,
)
from tqdm import tqdm
from transformers import HfArgumentParser

from prts.data.transforms import ImageTransforms, ImageTransformsConfig
from prts.training.config import TrainConfig
from prts.data.schema import LerobotConfig, DataConfig
from prts.data.normalize import load_norm_stats, calculate_dataset_statistics, cast_stats_to_numpy, LE_ROBOT_STATS_FILENAME, LE_ROBOT_DATA_FILENAME

"""lerobot datasets"""
class LeRobotDataset(BaseLeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        # custom features
        select_state_keys: list[str] | None = None,
        select_action_keys: list[str] | None = None,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
        )

        # remove unused features for efficiency
        self.set_feature_keys(select_state_keys, select_action_keys)

        # load dataset norm stats from pre-computed json file
        self.load_dataset_norm_stats()

    def load_dataset_norm_stats(self):
        # load dataset norm stats from pre-computed json file
        # if failed, compute stats and save as json
        import json
        norm_stats_path = Path(self.root) / LE_ROBOT_STATS_FILENAME
        if norm_stats_path.exists():
            norm_stats = load_norm_stats(self.root)
        else:
            print(f"Failed to load dataset statistics for {self.root.name}")
            print(f"Calculating dataset statistics for {self.root.name}")
            # Get all parquet files in the dataset paths
            parquet_files = list((self.root).glob(LE_ROBOT_DATA_FILENAME))
            select_keys = self.select_action_keys + self.select_state_keys
            norm_stats = calculate_dataset_statistics(select_keys, parquet_files)
            with open(norm_stats_path, "w") as f:
                json.dump(norm_stats, f, indent=4)
            norm_stats = cast_stats_to_numpy(norm_stats)

        q_existence = norm_stats[self.select_action_keys[0]]['q01'] is not None and \
            norm_stats[self.select_action_keys[0]]['q99'] is not None
        if not q_existence:
            raise ValueError("Quantile Normalization factors missing!")


    def set_feature_keys(self, state_keys=None, action_keys=None):
        """select state and action keys from the dataset"""
        self.select_state_keys = state_keys or [x for x in self.meta.features if x.startswith(OBS_STATE)]
        self.select_action_keys = action_keys or [x for x in self.meta.features if x.startswith(ACTION)]

class MultiLeRobotDataset(BaseMultiLeRobotDataset):
    def __init__(
        self,
        data_configs: list[LerobotConfig],
        image_transforms: Callable | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        chunk_size: int = 32,
    ):
        self.data_configs = data_configs
        self.chunk_size = chunk_size
        self.disabled_features = set()

        print(f"* load {len(data_configs)} lerobot datasets sequentially ...")
        datasets = []
        for idx in tqdm(range(len(data_configs)), desc="Loading lerobot datasets"):
            ds = _load_single_lerobot_dataset(
                idx,
                data_configs=self.data_configs,
                image_transforms=image_transforms,
                download_videos=download_videos,
                video_backend=video_backend,
                chunk_size=chunk_size,
            )
            if ds is not None:
                datasets.append(ds)

        self._datasets = [ds for ds in datasets if ds is not None]
        self.repo_ids = [ds.repo_id for ds in self._datasets]
        print(f"successfully load dataset {len(self.repo_ids)}/{len(data_configs)}:\n{self.repo_ids} ")

def _load_single_lerobot_dataset(
    idx,
    data_configs: list[LerobotConfig],
    image_transforms: Callable | None = None,
    download_videos: bool = True,
    video_backend: str | None = None,
    chunk_size: int = 50, # 32,
):
    """load a single lerobot dataset"""
    try:
        data_config = data_configs[idx]
        data_path = Path(data_config.root) if data_config.root else HF_LEROBOT_HOME / data_config.repo_id
        meta = LeRobotDatasetMetadata(data_config.repo_id, data_path)
        select_action_keys = data_config.select_action_keys or [
            k for k in meta.features if k.startswith(ACTION)
        ]
        delta_timestamps = {k: [i / meta.fps for i in range(0, chunk_size)] for k in select_action_keys}
        dataset = LeRobotDataset(
            data_config.repo_id,
            root=data_path,
            episodes=data_config.episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            download_videos=download_videos,
            video_backend=video_backend,
            select_state_keys=data_config.select_state_keys,
            select_action_keys=data_config.select_action_keys,
            # weight=data_config.weight,
        )
    except Exception as e:
        print(e)
        return None
    return dataset

def compute(args):
    data_configs = DataConfig.from_yaml(args.data_path)
    # load lerobot datasets
    if len(data_configs.lerobot_datasets) > 0:
        lerobot_dataset = MultiLeRobotDataset(
            data_configs=data_configs.lerobot_datasets,
            image_transforms=ImageTransforms(ImageTransformsConfig()),
            video_backend=args.lerobot_data_video_backend,
        )

if __name__ == "__main__":
    parser = HfArgumentParser(TrainConfig)
    (args,) = parser.parse_args_into_dataclasses()
    args.report_to = None
    compute(args)

