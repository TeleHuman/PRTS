import os
import logging
import time
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import transformers
from tqdm import tqdm
from functools import partial
from accelerate.logging import get_logger
from typing import List, Dict, Any, Tuple, Optional, Iterator, Callable
from colorlog import ColoredFormatter
import bisect


logger = get_logger(__name__, log_level="INFO")

LOG_FORMATTER = ColoredFormatter(
    fmt="%(log_color)s[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s]%(reset)s %(message_log_color)s%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        "DEBUG":    "cyan",
        "INFO":     "green",
        "WARNING":  "yellow",
        "ERROR":    "red",
        "CRITICAL": "bold_red",
    },
    secondary_log_colors={
        'message': {
            'INFO': 'white',
            'ERROR': 'red',
        }
    },
    style='%'
)

if not logger.logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(LOG_FORMATTER)
    logger.logger.addHandler(handler)


class EfficientLengthStorage:
    def __init__(self, mm_lengths: np.ndarray, lerobot_segments: List[Tuple[int, int, int]]):
        """
        Args:
            mm_lengths: mm_dataset的长度数组
            lerobot_segments: [(start_idx, end_idx, length), ...] 格式的lerobot分段
        """
        # mm_dataset部分：使用紧凑的numpy数组
        self.mm_lengths = mm_lengths.astype(np.uint16) 
        
        # lerobot_dataset部分：使用run-length encoding
        self.segment_starts = np.array([seg[0] for seg in lerobot_segments], dtype=np.uint32)
        self.segment_ends = np.array([seg[1] for seg in lerobot_segments], dtype=np.uint32)
        self.segment_lengths = np.array([seg[2] for seg in lerobot_segments], dtype=np.uint32)
        
        self.mm_count = len(mm_lengths)
        if self.segment_ends.shape[0] > 0:
            self.total_count = self.mm_count + (self.segment_ends[-1] - self.segment_starts[0] + 1)
        else:
            self.total_count = self.mm_count

    def __getitem__(self, idx: int) -> int:
        """O(1) for mm_dataset, O(log n) for lerobot_dataset"""
        if idx < self.mm_count:
            return int(self.mm_lengths[idx])
        else:
            # 在lerobot分段中二分查找
            pos = bisect.bisect_right(self.segment_starts, idx) - 1
            if pos >= 0 and self.segment_starts[pos] <= idx <= self.segment_ends[pos]:
                return int(self.segment_lengths[pos])
            raise IndexError(f"Index {idx} out of range")

    def __len__(self):
        return self.total_count

    def save_to_disk(self, path: str):
        assert self.mm_lengths is not None, "mm_lengths is None"
        # assert self.segment_starts is not None, "segment_starts is None"
        # assert self.segment_ends is not None, "segment_ends is None"
        # assert self.segment_lengths is not None, "segment_lengths is None"
        np.savez_compressed(path,
                        mm_lengths=self.mm_lengths,
                        segment_starts=self.segment_starts,
                        segment_ends=self.segment_ends,
                        segment_lengths=self.segment_lengths)

    @classmethod
    def load_from_disk(cls, path: str):
        data = np.load(path)
        return cls(
            mm_lengths=data['mm_lengths'],
            lerobot_segments=list(zip(
                data['segment_starts'],
                data['segment_ends'],
                data['segment_lengths']
            ))
        )


class LeRobotStylePacker:
    """
    参考LeRobot Dataset Format设计的分包数据管理器
    使用Parquet格式存储分包结果，支持高效索引访问和内存映射
    """
    
    def __init__(self, parquet_path: str = None, metadata: Dict[str, Any] = None):
        """
        Args:
            parquet_path: Parquet文件路径，如果提供则从文件加载
            meta 元数据字典，包含分包配置信息
        """
        self.parquet_path = parquet_path
        self.metadata = metadata or {}
        self._table = None
        self._indices_cache = {}
        
        if parquet_path and os.path.exists(parquet_path):
            self.load_from_parquet()
    
    def save(self, 
                     result: List[List[int]],
                     lengths: EfficientLengthStorage,
                     max_length: int, 
                     output_path: str,
                     seed: int = 42,
                     metadata: Dict[str, Any] = None):
        """
        执行分包并将结果保存为LeRobot风格的Parquet格式
        
        参考LeRobot的设计，所有分包元数据统一存储在结构化的Parquet文件中 [[3]]
        """
        logger.info("Starting saving process...")
        
        # 构建元数据
        self.metadata = metadata or {}
        self.metadata.update({
            'total_packages': len(result),
            'total_data_points': len(lengths),
            'max_length_per_package': max_length,
            'packing_seed': seed,
            'packing_algorithm': 'Worst_Fit_Decreasing',
            'creation_timestamp': pd.Timestamp.now().isoformat(),
            'statistics': {
                'avg_package_size': np.mean([len(pkg) for pkg in result]),
                'max_package_size': max(len(pkg) for pkg in result),
                'min_package_size': min(len(pkg) for pkg in result)
            }
        })
        
        # 转换为Parquet格式
        logger.info("Converting to Parquet format...")
        table = self._convert_to_arrow_table(result)
        
        # 保存Parquet文件
        self.parquet_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pq.write_table(table, output_path, compression='snappy')
        
        # 保存元数据文件（参考LeRobot的元数据管理方式 [[7]]）
        metadata_path = os.path.splitext(output_path)[0] + '_metadata.json'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Successfully saved {len(result)} packages to {output_path}")
        logger.info(f"Metadata saved to {metadata_path}")
        
        # 加载到内存映射
        self._table = pq.read_table(output_path, memory_map=True)
        return result
    
    def _convert_to_arrow_table(self, result: List[List[int]]) -> pa.Table:
        """
        将分包结果转换为Arrow Table，参考LeRobot的schema设计
        """
        # 创建列数据
        package_ids = list(range(len(result)))
        package_sizes = [len(pkg) for pkg in result]
        indices_data = [','.join(map(str, pkg)) for pkg in result]  # 逗号分隔的字符串
        
        # 构建Arrow schema（参考LeRobot的结构化设计 [[6]]）
        schema = pa.schema([
            ('package_id', pa.int32()),
            ('package_size', pa.int32()),
            ('indices_str', pa.string()),
            ('indices_count', pa.int32())
        ])
        
        # 创建表
        table = pa.table([
            pa.array(package_ids, type=pa.int32()),
            pa.array(package_sizes, type=pa.int32()),
            pa.array(indices_data, type=pa.string()),
            pa.array(package_sizes, type=pa.int32())
        ], schema=schema)
        
        return table
    
    def load_from_parquet(self, parquet_path: str = None):
        """
        从Parquet文件加载数据，使用内存映射提高性能
        """
        if parquet_path:
            self.parquet_path = parquet_path
        
        if not self.parquet_path or not os.path.exists(self.parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")
        
        logger.info(f"Loading data from {self.parquet_path} with memory mapping...")
        self._table = pq.read_table(self.parquet_path, memory_map=True)
        
        # 加载元数据
        metadata_path = os.path.splitext(self.parquet_path)[0] + '_metadata.json'
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        logger.info(f"Loaded {len(self)} packages successfully")
    
    def __len__(self) -> int:
        """返回包的总数"""
        if self._table is None:
            return 0
        return len(self._table)
    
    def __getitem__(self, package_idx: int) -> List[int]:
        """
        高速索引访问，返回指定包的索引列表
        使用缓存机制提高重复访问性能
        """
        if self._table is None:
            raise RuntimeError("No data loaded. Call load_from_parquet() first.")
        
        if package_idx < 0 or package_idx >= len(self):
            raise IndexError(f"Package index {package_idx} out of range [0, {len(self)-1}]")
        
        # 检查缓存
        if package_idx in self._indices_cache:
            return self._indices_cache[package_idx]
        
        # 从Parquet读取
        indices_str = self._table['indices_str'][package_idx].as_py()
        indices_list = list(map(int, indices_str.split(',')))
        
        # 缓存结果（小包缓存，大包不缓存以节省内存）
        if len(indices_list) < 1000:  # 阈值可根据需要调整
            self._indices_cache[package_idx] = indices_list
        
        return indices_list
    
    def __iter__(self) -> Iterator[List[int]]:
        """迭代所有包"""
        for i in range(len(self)):
            yield self[i]
    
    def get_package_info(self, package_idx: int) -> Dict[str, Any]:
        """获取指定包的详细信息"""
        if self._table is None:
            raise RuntimeError("No data loaded.")
        
        if package_idx < 0 or package_idx >= len(self):
            raise IndexError(f"Package index {package_idx} out of range")
        
        return {
            'package_id': self._table['package_id'][package_idx].as_py(),
            'package_size': self._table['package_size'][package_idx].as_py(),
            'indices_count': self._table['indices_count'][package_idx].as_py()
        }
    
    def get_batch(self, package_indices: List[int]) -> List[List[int]]:
        """
        批量获取多个包，比单个获取更高效
        """
        return [self[idx] for idx in package_indices]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        if self._table is None:
            return {}
        
        package_sizes = self._table['package_size'].to_pandas().values
        return {
            'total_packages': len(self),
            'total_indices': np.sum(package_sizes),
            'avg_package_size': np.mean(package_sizes),
            'max_package_size': np.max(package_sizes),
            'min_package_size': np.min(package_sizes),
            'std_package_size': np.std(package_sizes)
        }
    
    def export_to_hf_dataset(self):
        """
        导出为Hugging Face Dataset格式，与LeRobot生态系统集成 [[8]]
        """
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError("Please install datasets library: pip install datasets")
        
        if self._table is None:
            raise RuntimeError("No data loaded.")
        
        # 转换为pandas DataFrame
        df = self._table.to_pandas()
        
        # 创建HF Dataset
        hf_dataset = Dataset.from_pandas(df)
        return hf_dataset
    
    def validate_integrity(self) -> bool:
        """
        验证数据完整性，参考LeRobot的质量控制机制
        """
        if self._table is None:
            return False
        
        try:
            # 检查基本完整性
            total_packages = len(self)
            if total_packages == 0:
                logger.warning("No packages found")
                return False
            
            # 随机抽样验证
            import random
            sample_indices = random.sample(range(total_packages), min(10, total_packages))
            
            for idx in sample_indices:
                indices = self[idx]
                if not isinstance(indices, list) or len(indices) == 0:
                    logger.warning(f"Invalid package at index {idx}")
                    return False
            
            logger.info("Integrity check passed")
            return True
            
        except Exception as e:
            logger.error(f"Integrity check failed: {e}")
            return False
    
    def merge_with_another(self, other_packer: 'LeRobotStylePacker', output_path: str):
        """
        合并两个分包数据集，参考LeRobot的大规模数据集合并策略 [[1]]
        """
        if self._table is None or other_packer._table is None:
            raise RuntimeError("Both packers must have loaded data")
        
        logger.info("Merging two packed datasets...")
        
        # 获取所有包
        all_packages = []
        for i in range(len(self)):
            all_packages.append(self[i])
        for i in range(len(other_packer)):
            all_packages.append(other_packer[i])
        
        # 重新构建元数据
        merged_metadata = {
            'merged_from': [
                self.metadata.get('source', 'unknown'),
                other_packer.metadata.get('source', 'unknown')
            ],
            'total_packages': len(all_packages),
            'total_data_points': sum(len(pkg) for pkg in all_packages),
            'merge_timestamp': pd.Timestamp.now().isoformat(),
            'merge_strategy': 'concatenate'
        }
        
        # 保存为新的Parquet文件
        temp_packer = LeRobotStylePacker()
        temp_packer.metadata = merged_metadata
        
        # 转换为Arrow表
        merged_table = temp_packer._convert_to_arrow_table(all_packages)
        
        # 保存
        pq.write_table(merged_table, output_path, compression='snappy')
        
        # 保存元数据
        metadata_path = os.path.splitext(output_path)[0] + '_metadata.json'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(merged_metadata, f, indent=2)
        
        logger.info(f"Successfully merged datasets to {output_path}")
        return LeRobotStylePacker(output_path, merged_metadata)
    

def aggregate_dataset_length(dataset):
    """Aggregate the lengths of the dataset, used for dataset packing."""
    from torch.utils.data import DataLoader, Dataset

    total_data_len = len(dataset)
    mm_dataset_len = len(dataset.mm_dataset)

    num_workers = int(os.environ.get("DATASET_NUM_PROCESSES", 8))
    logger.info(f"Using {num_workers} workers to aggregate lengths ...")

    # Compute fixed lengths for lerobot sub-datasets using only the first item of each
    if dataset.lerobot_dataset:
        cumulative_sizes = dataset.lerobot_dataset.cumulative_sizes
    else:
        cumulative_sizes = []

    # Fill multimodal dataset lengths, using cached seq_length when available
    mm_lengths = np.zeros(mm_dataset_len, dtype=np.uint32)
    missing_mm_indices = []

    for i in tqdm(range(mm_dataset_len), desc="Aggregating MM lengths 1st loop"):
        num_tokens = dataset.mm_dataset.fetch_num_tokens(i)
        if num_tokens is not None:
            mm_lengths[i] = num_tokens
        else:
            missing_mm_indices.append(i)

    # For the remaining multimodal items without seq_length, parallelize a lightweight read
    if len(missing_mm_indices) > 0:
        class _IndexLengthDataset(Dataset):
            def __init__(self, base_dataset, indices):
                self.base_dataset = base_dataset
                self.indices = indices
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, j):
                idx = self.indices[j]
                return idx, int(self.base_dataset[idx]["input_ids"].shape[-1])

        def _collate_fn(batch):
            # batch: List[Tuple[idx, length]]
            idxs = [b[0] for b in batch]
            lens = [b[1] for b in batch]
            return idxs, lens

        # Avoid enabling action sampling while computing lengths
        original_sample_actions = getattr(dataset.mm_dataset, "sample_actions", None)
        try:
            if original_sample_actions is not None:
                dataset.mm_dataset.sample_actions = False
            subset = _IndexLengthDataset(dataset, missing_mm_indices)
            loader = DataLoader(
                subset,
                batch_size=max(32, 1),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True, # False,
                collate_fn=_collate_fn,
                persistent_workers=(num_workers > 0),
            )
            for idxs, lens in tqdm(loader, desc="Aggregating MM lengths 2nd loop"):
                for idx, l in zip(idxs, lens):
                    mm_lengths[idx] = int(l)
        finally:
            if original_sample_actions is not None:
                dataset.mm_dataset.sample_actions = original_sample_actions
            # [新增防僵尸补丁] 显式销毁临时 DataLoader，并强制释放共享内存
            if 'loader' in locals():
                # 强行关闭持久化 Worker 的内部迭代器
                if hasattr(loader, '_iterator') and loader._iterator is not None:
                    loader._iterator._shutdown_workers()
                del loader
            if 'subset' in locals():
                del subset
            import gc
            gc.collect()

    # Fill lerobot lengths in contiguous blocks to avoid per-item indexing
    lerobot_segments = []
    prev_cum = 0
    mm_offset = mm_dataset_len
    
    for ds_idx, cum_size in enumerate(cumulative_sizes):
        block_size = cum_size - prev_cum
        if block_size > 0:
            start_idx = mm_offset + prev_cum
            end_idx = mm_offset + cum_size - 1
            length_val = int((len(dataset[start_idx]["input_ids"]) + len(dataset[end_idx]["input_ids"])) / 2 )
            lerobot_segments.append((start_idx, end_idx, length_val))
        prev_cum = cum_size
    
    # 3. 创建高效存储结构
    lengths = EfficientLengthStorage(mm_lengths, lerobot_segments)

    # Finalize
    dataset.cached_lengths = lengths
    dataset.__setattr__("cached_lengths", lengths)
    assert len(lengths) == total_data_len, f"Length mismatch: {len(lengths)} != {total_data_len}"
    return lengths


def pack_data_points_by_length_with_shuffle(
    lengths: List[int], max_length: int, seed: int = 42, chunk_size: int = 1000000
) -> List[List[int]]:
    """
    Memory-optimized version: Shuffles the lengths before packing, while maintaining index mapping.
    Uses chunk processing and generators to reduce memory usage.

    Args:
        lengths (List[int]): List of lengths of data points.
        max_length (int): The concatenated length must be less than or equal to max_length.
        seed (int): Random seed for reproducible shuffling.
        chunk_size (int): Size of each chunk for processing, used to control memory usage.

    Returns:
        List[List[int]]: Groups of shuffled indices, e.g. [[shuffled_index1, shuffled_index2, ...], [], ...]
    """
    import random
    import numpy as np
    
    dataset_size = len(lengths)
    
    # 对于小数据集，使用原始方法
    if dataset_size <= chunk_size:
        return _pack_small_dataset_with_shuffle(lengths, max_length, seed)
    
    # 对于大数据集，使用内存优化方法
    logger.info(f"Using memory-optimized shuffle packing for large dataset (size: {dataset_size:,})")
    
    # 使用numpy的随机数生成器，更高效
    rng = np.random.RandomState(seed)
    
    # 分块处理
    result = []
    current_concatenated_length = 0
    current_list = []
    
    # 生成打乱的索引序列，但不全部存储在内存中
    shuffled_indices = rng.permutation(dataset_size)
    
    for i in tqdm(range(dataset_size), desc="Packing shuffled data points by length"):
        original_idx = shuffled_indices[i]
        cur_length = lengths[original_idx]
        
        if cur_length + current_concatenated_length <= max_length:
            current_concatenated_length += cur_length
            current_list.append(original_idx)
        else:  # current_list is done, create a new one
            if len(current_list) > 0:
                result.append(current_list)
            current_list = [original_idx]
            current_concatenated_length = cur_length

    if len(current_list) > 0:
        result.append(current_list)

    # assert to make sure no indices were missing
    assert sum([len(group) for group in result]) == dataset_size
    
    return result

def _pack_small_dataset_with_shuffle(lengths: List[int], max_length: int, seed: int = 42) -> List[List[int]]:
    """
    小数据集的shuffle packing实现（原始方法）
    """
    import random
    
    # 创建索引列表并打乱
    indices = list(range(len(lengths)))
    random.seed(seed)
    random.shuffle(indices)
    
    # 对打乱后的lengths进行packing
    result = []
    current_concatenated_length = 0
    current_list = []
    
    for i in range(len(indices)):
        original_idx = indices[i]
        cur_length = lengths[original_idx]
        
        if cur_length + current_concatenated_length <= max_length:
            current_concatenated_length += cur_length
            current_list.append(original_idx)
        else:  # current_list is done, create a new one
            if len(current_list) > 0:
                result.append(current_list)
            current_list = [original_idx]
            current_concatenated_length = cur_length

    if len(current_list) > 0:
        result.append(current_list)

    return result

import numpy as np
import numba
import time
import random
from typing import List, Tuple

# ==========================================
# Part 1: Numba 核心算法 (保持不变，极致性能)
# ==========================================

@numba.njit(fastmath=True)
def push_heap(heap, heap_size, bin_idx, bin_remains):
    """将 bin_idx 推入堆，基于 bin_remains 排序 (最大堆)"""
    i = heap_size
    heap[i] = bin_idx
    while i > 0:
        p = (i - 1) // 2
        if bin_remains[heap[i]] > bin_remains[heap[p]]:
            heap[i], heap[p] = heap[p], heap[i]
            i = p
        else:
            break

@numba.njit(fastmath=True)
def update_heap_root(heap, heap_size, bin_remains):
    """修改了堆顶元素的剩余空间后，执行下沉操作维持堆性质"""
    i = 0
    while True:
        left = 2 * i + 1
        right = 2 * i + 2
        largest = i
        
        if left < heap_size and bin_remains[heap[left]] > bin_remains[heap[largest]]:
            largest = left
        if right < heap_size and bin_remains[heap[right]] > bin_remains[heap[largest]]:
            largest = right
        
        if largest != i:
            heap[i], heap[largest] = heap[largest], heap[i]
            i = largest
        else:
            break

@numba.njit(parallel=False, fastmath=True)
def bin_pack_core(values, capacity):
    """
    核心装箱逻辑 (Worst-Fit Decreasing)
    """
    n = len(values)
    item_bin_assignments = np.empty(n, dtype=np.int32)
    bin_remains = np.zeros(n, dtype=np.int64) 
    
    # 手写堆
    heap = np.empty(n, dtype=np.int32)
    heap_size = 0
    bin_count = 0
    
    for i in range(n):
        val = values[i]
        
        # 异常处理：单个元素超过容量
        if val > capacity:
            bin_remains[bin_count] = 0 
            item_bin_assignments[i] = bin_count
            bin_count += 1
            continue

        # 检查堆顶
        if heap_size > 0:
            best_bin_idx = heap[0]
            if bin_remains[best_bin_idx] >= val:
                bin_remains[best_bin_idx] -= val
                item_bin_assignments[i] = best_bin_idx
                update_heap_root(heap, heap_size, bin_remains)
                continue
        
        # 开新桶
        new_bin_idx = bin_count
        bin_remains[new_bin_idx] = capacity - val
        item_bin_assignments[i] = new_bin_idx
        bin_count += 1
        
        push_heap(heap, heap_size, new_bin_idx, bin_remains)
        heap_size += 1
        
    return item_bin_assignments, bin_count

# ==========================================
# Part 2: 优化后的驱动函数 (引入 Megabatch)
# ==========================================

def pack_data_points_by_length_with_WFD(
    lengths: List[int], 
    max_length: int, 
    megabatch_size: int = 100000
) -> List[List[int]]:
    """
    针对 VLAM 训练优化的分块装箱策略。
    
    Args:
        lengths: 数据长度列表
        max_length: Bin 的最大容量
        megabatch_size: 宏批次大小。建议设为 global_batch_size 的 50-100 倍。
                        在这个范围内进行 Sort & Pack，既能减少 Padding，
                        又能保证整体数据的随机分布。
    
    Returns:
        List[List[int]]: 包含原始索引的 Bins 列表
    """
    print(f"Starting Megabatch Bin Packing (Total items: {len(lengths)})...")
    t0 = time.time()
    
    # 1. 预处理：转为 Numpy 数组以便快速索引
    # 使用 int32 节省内存，如果 idx 超过 21 亿请改 int64
    all_lengths = np.array(lengths, dtype=np.int32)
    num_samples = len(all_lengths)
    
    # 2. 全局 Shuffle：生成随机索引序列
    # 这是防止数据扎堆的关键步骤
    global_indices = np.arange(num_samples, dtype=np.int32)
    np.random.shuffle(global_indices)
    
    final_result_bins = []
    
    # 3. 遍历 Megabatch
    # 将数据切分成若干块，在块内进行 WFD 装箱
    num_chunks = (num_samples + megabatch_size - 1) // megabatch_size
    
    print(f"Processing {num_chunks} megabatches (Size: {megabatch_size})...")
    
    for i in range(0, num_samples, megabatch_size):
        # A. 获取当前块的全局索引
        chunk_global_indices = global_indices[i : i + megabatch_size]
        
        # B. 获取对应长度
        chunk_lengths = all_lengths[chunk_global_indices]
        
        # C. 块内排序 (Local Sort for WFD)
        # argsort 返回的是相对于 chunk_lengths 的索引 (0 ~ megabatch_size-1)
        local_sorted_args = np.argsort(-chunk_lengths) # 降序
        sorted_chunk_lengths = chunk_lengths[local_sorted_args]
        
        # D. Numba 核心计算
        # assignments: 每个物品(排序后)分配到的 bin_id
        assignments, num_bins_in_chunk = bin_pack_core(sorted_chunk_lengths, max_length)
        
        # E. 结果还原
        # 我们需要构建: Bin ID -> [Global Indices]
        # 这一步仍是 Python 循环，但因为是分块处理，内存压力较小
        current_chunk_bins = [[] for _ in range(num_bins_in_chunk)]
        
        # 预计算: 排序后的数据对应的 全局索引
        # sorted_global_indices[k] 就是第 k 个被放入 WFD 的物品的真实全局索引
        sorted_global_indices = chunk_global_indices[local_sorted_args]
        
        assignments_list = assignments.tolist()
        sorted_global_indices_list = sorted_global_indices.tolist()
        
        for global_idx, bin_id in zip(sorted_global_indices_list, assignments_list):
            current_chunk_bins[bin_id].append(global_idx)
            
        final_result_bins.extend(current_chunk_bins)

    # 4. 最终 Shuffle
    # 打乱 Bin 的顺序，防止同一个 Megabatch 的数据在训练时连续出现
    random.shuffle(final_result_bins)
    
    t_end = time.time()
    print(f"Packing complete. Total bins: {len(final_result_bins)}")
    print(f"Total time: {t_end - t0:.2f}s")
    
    return final_result_bins

def pack_data_points_by_length_with_shuffle_ultra_memory_efficient(
    lengths: List[int], max_length: int, seed: int = 42, chunk_size: int = 1000000
) -> List[List[int]]:
    """
    极致内存优化版本：适用于超大数据集（数十亿级别）
    使用分块shuffle + 临时文件来避免内存爆炸
    
    Args:
        lengths (List[int]): List of lengths of data points
        max_length (int): the concatenated length must be less than or equal max_length
        seed (int): 随机种子，用于可重现的打乱
        chunk_size (int): 每个chunk的大小，控制内存使用
        
    Returns:
        List[List[int]]: groups of shuffled indices
    """
    import random
    import tempfile
    import os
    import pickle
    
    dataset_size = len(lengths)
    
    # 对于小数据集，直接使用普通方法
    if dataset_size <= chunk_size:
        return _pack_small_dataset_with_shuffle(lengths, max_length, seed)
    
    logger.info(f"Using ultra memory-efficient shuffle packing for huge dataset (size: {dataset_size:,})")
    
    random.seed(seed)
    
    # 分块shuffle策略：
    # 1. 将数据分成多个chunk
    # 2. 每个chunk内部shuffle
    # 3. 然后将所有chunk的结果合并并shuffle
    
    num_chunks = (dataset_size + chunk_size - 1) // chunk_size
    temp_files = []
    
    try:
        # 第一阶段：分块处理
        for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, dataset_size)
            
            # 创建当前chunk的索引并shuffle
            chunk_indices = list(range(start_idx, end_idx))
            random.shuffle(chunk_indices)
            
            # 将shuffle后的索引保存到临时文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_chunk_{chunk_idx}.pkl')
            with open(temp_file.name, 'wb') as f:
                pickle.dump(chunk_indices, f)
            temp_files.append(temp_file.name)
            temp_file.close()
        
        # 第二阶段：合并所有chunk并进行packing
        result = []
        current_concatenated_length = 0
        current_list = []
        
        # 创建一个生成器来依次读取所有chunk的数据
        def merged_shuffled_indices():
            # 首先shuffle chunk的顺序
            chunk_order = list(range(num_chunks))
            random.shuffle(chunk_order)
            
            for chunk_idx in chunk_order:
                temp_file = temp_files[chunk_idx]
                with open(temp_file, 'rb') as f:
                    chunk_indices = pickle.load(f)
                for idx in chunk_indices:
                    yield idx
        
        # 进行packing
        for original_idx in tqdm(merged_shuffled_indices(), total=dataset_size, 
                                desc="Ultra memory-efficient packing"):
            cur_length = lengths[original_idx]
            
            if cur_length + current_concatenated_length <= max_length:
                current_concatenated_length += cur_length
                current_list.append(original_idx)
            else:  # current_list is done, create a new one
                if len(current_list) > 0:
                    result.append(current_list)
                current_list = [original_idx]
                current_concatenated_length = cur_length

        if len(current_list) > 0:
            result.append(current_list)
            
    finally:
        # 清理临时文件
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
    
    return result

def to_list_fast(storage: EfficientLengthStorage) -> list:
    # mm part: already dense
    mm_part = storage.mm_lengths.tolist()

    # lerobot part: expand segments in bulk
    lerobot_parts = []
    for start, end, length in zip(storage.segment_starts, storage.segment_ends, storage.segment_lengths):
        count = end - start + 1
        lerobot_parts.append(np.full(count, length, dtype=np.uint32))
    lerobot_part = np.concatenate(lerobot_parts) if lerobot_parts else np.array([], dtype=np.uint32)

    return mm_part + lerobot_part.tolist()


def validate_packing_efficiency(packed_bins: List[List[int]], lengths: List[int], capacity: int, sample_size: int = 100):
    """
    验证分包效果：抽检部分包，检查容量利用率和是否超限
    
    Args:
        packed_bins: 分包结果 [[indices], [indices], ...]
        lengths: 原始长度列表
        capacity: 包容量限制
        sample_size: 抽检样本数量
    """
    import random
    import numpy as np
    
    logger.info(f"🔍 开始分包效果验证抽检 (样本数: {sample_size})...")
    
    # 随机抽样
    sample_indices = random.sample(range(len(packed_bins)), min(sample_size, len(packed_bins)))
    utilization_rates = []
    overflow_count = 0
    
    print("\n=== 分包效果抽检报告 ===")
    print(f"总包数: {len(packed_bins)}, 抽检样本数: {len(sample_indices)}")
    print("-" * 60)
    
    for i, bin_idx in enumerate(sample_indices[:10]):  # 只显示前10个详细结果
        bin_items = packed_bins[bin_idx]
        total_length = sum(lengths[idx] for idx in bin_items)
        utilization = total_length / capacity * 100
        
        utilization_rates.append(utilization)
        
        if total_length > capacity:
            overflow_count += 1
            status = "❌ 超限"
        elif utilization >= 95:
            status = "✅ 优秀"
        elif utilization >= 85:
            status = "⭐ 良好"
        else:
            status = "⚠️  待优化"
        
        print(f"包 #{bin_idx:5d} | 物品数: {len(bin_items):3d} | 总长度: {total_length:6d}/{capacity} | 利用率: {utilization:5.1f}% | 状态: {status}")
    
    if len(sample_indices) > 10:
        print(f"... (其余 {len(sample_indices)-10} 个样本略)")
    
    # 计算统计指标
    avg_utilization = np.mean(utilization_rates)
    max_utilization = np.max(utilization_rates)
    min_utilization = np.min(utilization_rates)
    
    print("-" * 60)
    print(f"📈 利用率统计: 平均 {avg_utilization:.1f}%, 最高 {max_utilization:.1f}%, 最低 {min_utilization:.1f}%")
    print(f"🚨 超限包数: {overflow_count}/{len(sample_indices)}")
    
    # 评估分包质量
    if overflow_count > 0:
        quality = "❌ 不合格 - 存在超限包"
    elif avg_utilization >= 90:
        quality = "✅ 优秀 - 高效利用容量"
    elif avg_utilization >= 80:
        quality = "⭐ 良好 - 合理利用容量"
    else:
        quality = "⚠️  需要优化 - 容量利用率偏低"
    
    print(f"🎯 分包质量评估: {quality}")
    print("=" * 60)
    
    # 返回验证结果供程序使用
    return {
        'avg_utilization': avg_utilization,
        'overflow_count': overflow_count,
        'sample_size': len(sample_indices),
        'quality_rating': quality
    }


def pack_data_points_by_length_with_shuffle_streaming(
    lengths: List[int], max_length: int, seed: int = 42
) -> List[List[int]]:
    """
    流式处理版本：中等内存优化，适用于大数据集
    避免创建额外的shuffled_lengths数组
    
    Args:
        lengths (List[int]): List of lengths of data points
        max_length (int): the concatenated length must be less than or equal max_length
        seed (int): 随机种子，用于可重现的打乱
        
    Returns:
        List[List[int]]: groups of shuffled indices
    """
    import random
    
    dataset_size = len(lengths)
    random.seed(seed)
    
    result = []
    current_concatenated_length = 0
    current_list = []
    
    # 创建一个迭代器来生成打乱的索引
    def shuffled_indices_generator():
        indices = list(range(dataset_size))
        random.shuffle(indices)
        for idx in indices:
            yield idx
    
    for original_idx in tqdm(shuffled_indices_generator(), total=dataset_size, 
                            desc="Streaming shuffle packing"):
        cur_length = lengths[original_idx]
        
        if cur_length + current_concatenated_length <= max_length:
            current_concatenated_length += cur_length
            current_list.append(original_idx)
        else:  # current_list is done, create a new one
            if len(current_list) > 0:
                result.append(current_list)
            current_list = [original_idx]
            current_concatenated_length = cur_length

    if len(current_list) > 0:
        result.append(current_list)

    return result

def pack_data_points_by_length(
    lengths: List[int], max_length: int
) -> List[List[int]]:
    """given lengths of data points, we merge consecutive data points into a new data point, as long as the concatenated length is less than max_length
    Args:
        lengths (List[int]): List of lengths of data points
        max_length (int): the concatenated length must be less than or equal max_length
        max_size: if != -1; the maximum number of consecutive items being merged; max_size: -1 --> no limit for number of items being merged

    max_size: the maximum number of data points being merged
    For example, lengths=[1, 3, 2, 2, 6, 4, 2, 6, 5]; max_length=10
     --> [[0,1,2,3], [4, 5], [6,7], [8]]

    Returns:
        _type_: groups of indices: [[index1, index2, ...], [], ...]
    """
    result = []
    current_concatenated_length = 0
    current_list = []
    for i in tqdm(range(len(lengths)), desc="Packing data points by length"):
        cur_length = lengths[i]
        if cur_length + current_concatenated_length <= max_length:
            current_concatenated_length += cur_length
            current_list.append(i)
        else:  # current_list is done, create a new one
            if len(current_list) > 0:
                result.append(current_list)
            current_list = [i]
            current_concatenated_length = cur_length

    if len(current_list) > 0:
        result.append(current_list)

    # assert to make sure no indices were missing
    assert sum([len(indices) for indices in result]) == len(lengths)
    return result


#### related to the training
def set_requires_grad(parameters, requires_grad):
    """Set the requires_grad attribute for the parameters."""
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(vlm, training_args, compute_dtype, device):
    """Configure the vision tower."""
    vision_tower = vlm.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = vlm.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    merger_params = vlm.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)


def configure_llm(vlm, training_args):
    """Configure the LLM."""
    lm_head = vlm.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_lm_head)

    llm_params = vlm.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def configure_llm_qwen3_vl(vlm, training_args):
    """Configure the LLM."""
    lm_head = vlm.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_lm_head)

    llm_params = vlm.language_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


'''
for packing dataset
'''
try:
    # from flash_attn.flash_attn_interface import flash_attn_varlen_func    # FA2
    from flash_attn_interface import flash_attn_varlen_func # FA3
except ImportError:
    print("flash_attn is not installed")

from transformers.utils.deprecation import deprecate_kwarg
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import Cache
# from ..models.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
from ..models.modeling_qwen3_vl import apply_rotary_pos_emb

def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )
    
    # This is before the transpose
    seq_len = query.shape[2]

    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            "Tensor query has shape  with a zero dimension.\n"
            "FlashAttention does not support inputs with dim=0.\n"
            "Please check your input shapes or use SDPA instead."
        )

    # FA2 uses non-transposed inputs
    # batch, head, seq_len, dim
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    # batch, seqlen, head, dim

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype


    # Packed FA varlen with cu_seqlens
    query = query.squeeze(0)
    key = key.squeeze(0)
    value = value.squeeze(0)
    cu_seqlens = attention_mask

    max_seqlen = kwargs.get("max_seqlen")
    if max_seqlen is None:
        max_seqlen = getattr(module.config, "max_seqlen", None)
    if max_seqlen is None:
        with torch.no_grad():
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    else:
        max_seqlen = int(max_seqlen)

    attn_output = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )

    attn_output = attn_output.unsqueeze(0)

    return attn_output, None


@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen2vl_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output, attn_weights = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        position_ids=position_ids,  # pass positions for FA2
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen3vl_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    # DBEUG: In the `kwargs` here, `position_ids` is actually `text_position_ids`
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output, attn_weights = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights



def return_mask(
    config,
    input_embeds,
    attention_mask,
    cache_position,
    past_key_values,
    position_ids,
    **kwargs
):
    return attention_mask


def replace_qwen2_vl_attention_class():
    import prts.models.modeling_qwen2_5_vl as our_modeling_qwen2_5_vl
    import prts.models.modeling_qwen3_vl as our_modeling_qwen3_vl
    import transformers.modeling_flash_attention_utils

    ## qwen2_5_vl
    our_modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = (
        qwen2vl_forward
    )
    our_modeling_qwen2_5_vl.create_causal_mask = (
        return_mask
    )
    our_modeling_qwen2_5_vl.create_sliding_window_causal_mask = (
        return_mask
    )

    ## qwen3vl
    our_modeling_qwen3_vl.Qwen3VLTextAttention.forward = (
        qwen3vl_forward
    )
    our_modeling_qwen3_vl.create_causal_mask = (
        return_mask
    )
    ## qwen3vl moe
    # transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextAttention.forward = (
    #     qwen3vl_forward
    # )
    # transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.create_causal_mask = (
    #     return_mask
    # )

def set_dataset_local_rank(local_rank: int):
    import prts.data as _data
    _data.dataset.local_rank = local_rank
    _data.lerobot_dataset.local_rank = local_rank
    _data.multim_dataset.local_rank = local_rank
