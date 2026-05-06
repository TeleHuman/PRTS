import json
import pathlib

import numpy as np
import numpydantic
import pydantic
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol
import torch
from torch import nn, Tensor

from prts.utils.logging_utils import (
    normal,
    info,
    success,
    warning,
    error,
    fail,
    debug,
    important,
    cprint
)

# LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
# LE_ROBOT_STEPS_FILENAME = "meta/steps.pkl"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/norm_stats.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
# EPSILON = 5e-4

@pydantic.dataclasses.dataclass
class NormStats:
    # min: numpydantic.NDArray
    # max: numpydantic.NDArray
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile
    q99: numpydantic.NDArray | None = None  # 99th quantile

class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch: np.ndarray) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            vectors (np.ndarray): A 2D array where each row is a new vector.
        """
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """
        Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99)
        # return NormStats(min=self._min, max=self._max, mean=self._mean, std=stddev, q01=q01, q99=q99)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]

def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    def to_dict(v: NormStats) -> dict:
        return {
            # "min": v.min.tolist(),
            # "max": v.max.tolist(),
            "mean": v.mean.tolist(),
            "std": v.std.tolist(),
            "q01": v.q01.tolist() if v.q01 is not None else None,
            "q99": v.q99.tolist() if v.q99 is not None else None,
        }

    return json.dumps({k: to_dict(v) for k, v in norm_stats.items()}, indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())


# ---------------------------------------------------------------------------
# Saving / loading norm stats alongside a saved model
# ---------------------------------------------------------------------------

#: Filename used when saving norm stats next to model weights.
MODEL_NORM_STATS_FILENAME = "statistics.json"


def _stats_to_serializable(
    stats: dict[str, dict],
) -> dict[str, dict]:
    """Convert stat values (np.ndarray / torch.Tensor / list) to plain lists."""
    out = {}
    for key, key_stats in stats.items():
        out[key] = {}
        for stat_name, val in key_stats.items():
            if val is None:
                out[key][stat_name] = None
            elif isinstance(val, np.ndarray):
                out[key][stat_name] = val.tolist()
            elif isinstance(val, torch.Tensor):
                out[key][stat_name] = val.cpu().tolist()
            else:
                # already list / scalar
                out[key][stat_name] = val
    return out


def save_norm_stats_for_model(
    output_dir: pathlib.Path | str,
    datasets_stats: dict[str, dict],
    state_mode: str = "MEAN_STD",
) -> None:
    """Save norm stats alongside model weights for eval-time use.

    The file is written to ``output_dir/norm_stats.json`` and has the form::

        {
          "state_mode": "MEAN_STD",
          "features": {...},   # from the first dataset (default for eval)
          "stats":    {...},   # from the first dataset
          "datasets": {
            "repo/name": {"features": {...}, "stats": {...}},
            ...
          }
        }

    Args:
        output_dir: Model checkpoint directory (same folder as ``config.json``).
        datasets_stats: Mapping ``{repo_id: {"features": raw_features_dict,
            "stats": {key: {stat_name: array_or_list}}}}``.
        state_mode: Action normalisation mode string (e.g. ``"MEAN_STD"``).
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Top-level entry = first dataset (used as default during eval)
    first_entry = next(iter(datasets_stats.values())) if datasets_stats else {}

    payload: dict = {
        "state_mode": state_mode,
        "features": first_entry.get("features", {}),
        "stats": _stats_to_serializable(first_entry.get("stats", {})),
        "datasets": {
            repo_id: {
                "features": entry["features"],
                "stats": _stats_to_serializable(entry["stats"]),
            }
            for repo_id, entry in datasets_stats.items()
        },
    }

    out_path = output_dir / MODEL_NORM_STATS_FILENAME
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_normalizer_from_model_dir(
    model_dir: pathlib.Path | str,
    state_mode: str | None = None,
    repo_id: str | None = None,
    device: str = "cpu",
) -> tuple["Normalize", "Unnormalize"]:
    """Reconstruct normalizer/unnormalizer from ``norm_stats.json`` saved with the model.

    Args:
        model_dir: Path to the saved model directory.
        state_mode: Normalisation mode for actions (e.g. ``"MEAN_STD"``).
            If ``None``, the value stored in the file is used.
        repo_id: Load stats for this specific dataset. If ``None`` or not found,
            the top-level (first-dataset default) entry is used.
        device: PyTorch device string.

    Returns:
        ``(normalizer, unnormalizer)`` pair ready for inference.

    Raises:
        FileNotFoundError: If ``norm_stats.json`` is absent from ``model_dir``.
    """
    model_dir = pathlib.Path(model_dir)
    stats_path = model_dir / MODEL_NORM_STATS_FILENAME
    if not stats_path.exists():
        raise FileNotFoundError(
            f"'{MODEL_NORM_STATS_FILENAME}' not found in {model_dir}. "
            "Generate it by running training with PRTSTrainer, "
            "or pass dataset_path to PRTSInterface instead."
        )

    with open(stats_path) as f:
        payload = json.load(f)

    if state_mode is None:
        state_mode = payload.get("state_mode", "MEAN_STD")

    # Pick the right entry
    if repo_id is not None and repo_id in payload.get("datasets", {}):
        entry = payload["datasets"][repo_id]
    else:
        entry = payload  # top-level default (first dataset)

    raw_features: dict = entry.get("features", {})
    raw_stats: dict = entry.get("stats", {})

    # Convert list values → np.ndarray
    stats: dict[str, dict[str, np.ndarray]] = {}
    for key, key_stats in raw_stats.items():
        stats[key] = {
            sname: np.array(sval, dtype=np.float32)
            for sname, sval in key_stats.items()
            if sval is not None
        }

    # Reconstruct PolicyFeature objects from stored feature metadata
    features: dict[str, "PolicyFeature"] = {}
    for key, ft_info in raw_features.items():
        dtype_str: str = ft_info.get("dtype", "")
        shape: tuple = tuple(ft_info.get("shape", []))

        if dtype_str in ("image", "video"):
            feat_type = FeatureType.VISUAL
        elif key == "observation.environment_state":
            feat_type = FeatureType.ENV
        elif key.startswith("observation"):
            feat_type = FeatureType.STATE
        elif key.startswith("action"):
            feat_type = FeatureType.ACTION
        else:
            continue

        features[key] = PolicyFeature(type=feat_type, shape=shape)

    norm_map: dict[str, NormalizationMode] = {
        "STATE": NormalizationMode.QUANTILE,
        "ACTION": NormalizationMode(state_mode),
    }
    unnorm_map: dict[str, NormalizationMode] = {
        "ACTION": NormalizationMode(state_mode),
    }

    normalizer = Normalize(features, norm_map, stats).to(device)
    unnormalizer = Unnormalize(features, unnorm_map, stats).to(device)
    
    norm_info = {k: norm_map.get(ft.type, NormalizationMode.IDENTITY).value for k, ft in features.items()}
    unnorm_info = {k: unnorm_map.get(ft.type, NormalizationMode.IDENTITY).value for k, ft in features.items()}
    info(f"Normalizer modes:\n{json.dumps(norm_info, indent=2)}")
    info(f"Unnormalizer modes:\n{json.dumps(unnorm_info, indent=2)}")

    return normalizer, unnormalizer


from pathlib import Path
import json 
def cast_stats_to_numpy(stats) -> dict[str, dict[str, np.ndarray]]:
    stats = {key: np.array(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(stats)

def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dictionary structure by collapsing nested keys into one key with a separator.

    For example:
    ```
    >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}`
    >>> print(flatten_dict(dct))
    {"a/b": 1, "a/c/d": 2, "e": 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(d: dict, sep: str = "/") -> dict:
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict

def load_norm_stats(local_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    norm_stats_path = local_dir / LE_ROBOT_STATS_FILENAME
    if norm_stats_path.exists():
        with open(norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
        # success(f"Loading quantile action norm stats from {norm_stats_path}")
        return cast_stats_to_numpy(norm_stats)
    return None

import pyarrow.parquet as pq
import pandas as pd

def calculate_dataset_statistics(select_keys, parquet_paths: list[Path]) -> dict:
    """Calculate the dataset statistics of all columns for a list of parquet files using pyarrow."""
    all_low_dim_data_list = []

    # === Collect all parquet data ===
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # ✅ Use pyarrow to read Parquet
        table = pq.read_table(parquet_path)
        parquet_data = table.to_pandas(types_mapper=pd.ArrowDtype)

        all_low_dim_data_list.append(parquet_data)

    # ✅ Concatenate all tables
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0, ignore_index=True)

    # === Compute dataset statistics ===
    dataset_statistics = {}
    for le_modality in select_keys:
        info(f"Computing statistics for {le_modality}...")

        # ✅ Convert column safely to ndarray
        col_data = all_low_dim_data[le_modality]
        # Avoid object dtype and ArrowArray issues
        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in col_data if x is not None]
        )

        dataset_statistics[le_modality] = {
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }

    return dataset_statistics


### LeRobot style normalizer & unnormalizer
class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    REWARD = "REWARD"


class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"
    QUANTILE = "QUANTILE" # use q01 and q99

class DictLike(Protocol):
    def __getitem__(self, key: Any) -> Any: ...

@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple



def create_stats_buffers(
    features: dict[str, PolicyFeature],
    norm_map: dict[str, NormalizationMode],
    stats: dict[str, dict[str, Tensor]] | None = None,
) -> dict[str, dict[str, nn.ParameterDict]]:
    """
    Create buffers per modality (e.g. "observation.image", "action") containing their mean, std, min, max
    statistics.

    Args: (see Normalize and Unnormalize)

    Returns:
        dict: A dictionary where keys are modalities and values are `nn.ParameterDict` containing
            `nn.Parameters` set to `requires_grad=False`, suitable to not be updated during backpropagation.
    """
    stats_buffers = {}

    for key, ft in features.items():
        norm_mode = norm_map.get(ft.type, NormalizationMode.IDENTITY)
        if norm_mode is NormalizationMode.IDENTITY:
            continue

        assert isinstance(norm_mode, NormalizationMode)

        shape = tuple(ft.shape)

        if ft.type is FeatureType.VISUAL:
            # sanity checks
            assert len(shape) == 3, f"number of dimensions of {key} != 3 ({shape=}"
            c, h, w = shape
            assert c < h and c < w, f"{key} is not channel first ({shape=})"
            # override image shape to be invariant to height and width
            shape = (c, 1, 1)

        # Note: we initialize mean, std, min, max to infinity. They should be overwritten
        # downstream by `stats` or `policy.load_state_dict`, as expected. During forward,
        # we assert they are not infinity anymore.

        buffer = {}
        if norm_mode is NormalizationMode.MEAN_STD:
            mean = torch.ones(shape, dtype=torch.float32) * torch.inf
            std = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "mean": nn.Parameter(mean, requires_grad=False),
                    "std": nn.Parameter(std, requires_grad=False),
                }
            )
        elif norm_mode is NormalizationMode.MIN_MAX:
            min = torch.ones(shape, dtype=torch.float32) * torch.inf
            max = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "min": nn.Parameter(min, requires_grad=False),
                    "max": nn.Parameter(max, requires_grad=False),
                }
            )
        
        elif norm_mode is NormalizationMode.QUANTILE:
            q01 = torch.ones(shape, dtype=torch.float32) * torch.inf
            q99 = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "q01": nn.Parameter(q01, requires_grad=False),
                    "q99": nn.Parameter(q99, requires_grad=False),
                }
            )

        # TODO(aliberts, rcadene): harmonize this to only use one framework (np or torch)
        if stats:
            if isinstance(stats[key]["mean"], np.ndarray):
                if norm_mode is NormalizationMode.MEAN_STD:
                    buffer["mean"].data = torch.from_numpy(stats[key]["mean"]).to(dtype=torch.float32)
                    buffer["std"].data = torch.from_numpy(stats[key]["std"]).to(dtype=torch.float32)
                elif norm_mode is NormalizationMode.MIN_MAX:
                    buffer["min"].data = torch.from_numpy(stats[key]["min"]).to(dtype=torch.float32)
                    buffer["max"].data = torch.from_numpy(stats[key]["max"]).to(dtype=torch.float32)
                elif norm_mode is NormalizationMode.QUANTILE:
                    buffer["q01"].data = torch.from_numpy(stats[key]["q01"]).to(dtype=torch.float32)
                    buffer["q99"].data = torch.from_numpy(stats[key]["q99"]).to(dtype=torch.float32)
            elif isinstance(stats[key]["mean"], torch.Tensor):
                # Note: The clone is needed to make sure that the logic in save_pretrained doesn't see duplicated
                # tensors anywhere (for example, when we use the same stats for normalization and
                # unnormalization). See the logic here
                # https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/py_src/safetensors/torch.py#L97.
                if norm_mode is NormalizationMode.MEAN_STD:
                    buffer["mean"].data = stats[key]["mean"].clone().to(dtype=torch.float32)
                    buffer["std"].data = stats[key]["std"].clone().to(dtype=torch.float32)
                elif norm_mode is NormalizationMode.MIN_MAX:
                    buffer["min"].data = stats[key]["min"].clone().to(dtype=torch.float32)
                    buffer["max"].data = stats[key]["max"].clone().to(dtype=torch.float32)
                elif norm_mode is NormalizationMode.QUANTILE:
                    buffer["q01"].data = stats[key]["q01"].clone().to(dtype=torch.float32)
                    buffer["q99"].data = stats[key]["q99"].clone().to(dtype=torch.float32)
            else:
                type_ = type(stats[key]["mean"])
                raise ValueError(f"np.ndarray or torch.Tensor expected, but type is '{type_}' instead.")

        stats_buffers[key] = buffer
    return stats_buffers

def _no_stats_error_str(name: str) -> str:
    raise ValueError(
        f"`{name}` is infinity. You should either initialize with `stats` as an argument, or use a "
        "pretrained model."
    )


class Normalize(nn.Module):
    """Normalizes data (e.g. "observation.image") for more stable and faster convergence during training."""

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        """
        Args:
            shapes (dict): A dictionary where keys are input modalities (e.g. "observation.image") and values
            are their shapes (e.g. `[3,96,96]`]). These shapes are used to create the tensor buffer containing
            mean, std, min, max statistics. If the provided `shapes` contain keys related to images, the shape
            is adjusted to be invariant to height and width, assuming a channel-first (c, h, w) format.
            modes (dict): A dictionary where keys are output modalities (e.g. "observation.image") and values
                are their normalization modes among:
                    - "mean_std": subtract the mean and divide by standard deviation.
                    - "min_max": map to [-1, 1] range.
            stats (dict, optional): A dictionary where keys are output modalities (e.g. "observation.image")
                and values are dictionaries of statistic types and their values (e.g.
                `{"mean": torch.randn(3,1,1)}, "std": torch.randn(3,1,1)}`). If provided, as expected for
                training the model for the first time, these statistics will overwrite the default buffers. If
                not provided, as expected for finetuning or evaluation, the default buffers should to be
                overwritten by a call to `policy.load_state_dict(state_dict)`. That way, initializing the
                dataset is not needed to get the stats, since they are already in the policy state_dict.
        """
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats = stats
        stats_buffers = create_stats_buffers(features, norm_map, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    # TODO(rcadene): should we remove torch.no_grad?
    @torch.no_grad()
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # TODO: Remove this shallow copy
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        for key, ft in self.features.items():
            if key not in batch:
                # FIXME(aliberts, rcadene): This might lead to silent fail!
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = buffer["mean"]
                std = buffer["std"]
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                batch[key] = (batch[key] - mean) / (std + 1e-8)
            elif norm_mode is NormalizationMode.MIN_MAX:
                min = buffer["min"]
                max = buffer["max"]
                assert not torch.isinf(min).any(), _no_stats_error_str("min")
                assert not torch.isinf(max).any(), _no_stats_error_str("max")
                # normalize to [0,1]
                batch[key] = (batch[key] - min) / (max - min + 1e-8)
                # normalize to [-1, 1]
                batch[key] = batch[key] * 2 - 1
            
            elif norm_mode is NormalizationMode.QUANTILE:
                q01 = buffer["q01"]
                q99 = buffer["q99"]
                assert not torch.isinf(q01).any(), _no_stats_error_str("q01")
                assert not torch.isinf(q99).any(), _no_stats_error_str("q99")
                batch[key] = (batch[key] - q01) / (q99 - q01 + 1e-8)
                batch[key] = batch[key] * 2 - 1

            else:
                raise ValueError(norm_mode)
        return batch

class Unnormalize(nn.Module):
    """
    Similar to `Normalize` but unnormalizes output data (e.g. `{"action": torch.randn(b,c)}`) in their
    original range used by the environment.
    """

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            shapes (dict): A dictionary where keys are input modalities (e.g. "observation.image") and values
            are their shapes (e.g. `[3,96,96]`]). These shapes are used to create the tensor buffer containing
            mean, std, min, max statistics. If the provided `shapes` contain keys related to images, the shape
            is adjusted to be invariant to height and width, assuming a channel-first (c, h, w) format.
            modes (dict): A dictionary where keys are output modalities (e.g. "observation.image") and values
                are their normalization modes among:
                    - "mean_std": subtract the mean and divide by standard deviation.
                    - "min_max": map to [-1, 1] range.
            stats (dict, optional): A dictionary where keys are output modalities (e.g. "observation.image")
                and values are dictionaries of statistic types and their values (e.g.
                `{"mean": torch.randn(3,1,1)}, "std": torch.randn(3,1,1)}`). If provided, as expected for
                training the model for the first time, these statistics will overwrite the default buffers. If
                not provided, as expected for finetuning or evaluation, the default buffers should to be
                overwritten by a call to `policy.load_state_dict(state_dict)`. That way, initializing the
                dataset is not needed to get the stats, since they are already in the policy state_dict.
        """
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats = stats
        # `self.buffer_observation_state["mean"]` contains `torch.tensor(state_dim)`
        stats_buffers = create_stats_buffers(features, norm_map, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    # TODO(rcadene): should we remove torch.no_grad?
    @torch.no_grad()
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        for key, ft in self.features.items():
            if key not in batch:
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = buffer["mean"]
                std = buffer["std"]
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                batch[key] = batch[key] * std + mean
            elif norm_mode is NormalizationMode.MIN_MAX:
                min = buffer["min"]
                max = buffer["max"]
                assert not torch.isinf(min).any(), _no_stats_error_str("min")
                assert not torch.isinf(max).any(), _no_stats_error_str("max")
                batch[key] = (batch[key] + 1) / 2
                batch[key] = batch[key] * (max - min) + min

            elif norm_mode is NormalizationMode.QUANTILE:
                q01 = buffer["q01"]
                q99 = buffer["q99"]
                assert not torch.isinf(q01).any(), _no_stats_error_str("q01")
                assert not torch.isinf(q99).any(), _no_stats_error_str("q99")
                batch[key] = (batch[key] + 1) / 2
                batch[key] = batch[key] * (q99 - q01) + q01

            else:
                raise ValueError(norm_mode)
        return batch

