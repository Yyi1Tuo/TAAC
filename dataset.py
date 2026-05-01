"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.
"""

import os
import logging
import random
import json
import gc

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

# numpy.typing is available since numpy >= 1.20; on older numpy fall back to a
# no-op shim so that forward-referenced annotations like ``npt.NDArray[np.int64]``
# keep working as plain strings without raising at import time.
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.

    For int features:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int part length
    For dense features:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float part length
    """

    def __init__(self) -> None:
        # Ordered list of (feature_id, offset, length).
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        # Quick lookup from fid to its (offset, length).
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """Append a feature to the schema."""
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """Get ``(offset, length)`` for a feature_id."""
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        """Return all feature_ids in their insertion order."""
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for JSON dumping)."""
        return {
            'entries': self.entries,
            'total_dim': self.total_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        """Reconstruct a :class:`FeatureSchema` from its dict form."""
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)

# Use filesystem-based tensor sharing (instead of /dev/shm) to avoid running
# out of shared memory when many DataLoader workers are active.
torch.multiprocessing.set_sharing_strategy('file_system')

# Time-delta bucket boundaries (64 edges -> 65 buckets: 0=padding, 1..64).
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

# Total number of time-bucket embedding slots (= number of boundaries + 1, with
# padding=0 included).
#
# This constant is uniquely determined by the length of BUCKET_BOUNDARIES; on
# the model side, ``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` must match
# this value exactly, otherwise an IndexError may be raised at runtime.
#
# That is why ``train.py`` / ``infer.py`` only expose the boolean flag
# ``--use_time_buckets`` and derive the concrete bucket count from here.
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1
SEQ_TIME_STAT_FEATS_PER_DOMAIN = 6
ITEM_STRUCT_STAT_FEATS = 7


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    - int features: scalar or list (multi-hot); values <= 0 are mapped to 0 (padding).
    - dense features: ``list<float>``, variable-length padded up to ``max_dim``.
    - sequence features: ``list<int64>``, grouped by domain; includes side-info
      columns and an optional timestamp column (used for time-bucketing).
    - label: mapped from ``label_type == 2``.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        ts_filter: Optional[Tuple[Optional[int], Optional[int]]] = None,
    ) -> None:
        """
        Args:
            parquet_path: either a directory containing ``*.parquet`` files or
                a single parquet file path.
            schema_path: path of the schema JSON describing feature layouts.
            batch_size: fixed batch size used for the pre-allocated buffers.
            seq_max_lens: optional per-domain override of sequence truncation,
                e.g. ``{'seq_d': 256}``. Domains not listed fall back to the
                schema default of 256.
            shuffle: whether to shuffle within a ``buffer_batches``-sized window.
            buffer_batches: shuffle buffer size in units of batches.
            row_group_range: ``(start, end)`` slice of Row Groups; ``None`` to
                use all Row Groups.
            clip_vocab: if True, clip out-of-bound ids to 0; if False, raise.
            is_training: if True, derive ``label`` from ``label_type == 2``;
                if False, return an all-zeros label column.
            ts_filter: optional ``(min_ts, max_ts)`` half-open range. Only
                rows whose ``timestamp`` satisfies ``min_ts <= ts < max_ts``
                are kept; either bound can be ``None`` (= unbounded). This
                is the row-level time split used to fight train/valid
                temporal leakage when Row-Group-level splits are too coarse.
        """
        super().__init__()
        # Row-level timestamp filter (applied at the end of every batch
        # in ``_convert_batch``). Stored as plain attributes so the
        # iterator can read them without dict lookups.
        if ts_filter is not None:
            self._ts_filter_min, self._ts_filter_max = ts_filter
        else:
            self._ts_filter_min, self._ts_filter_max = None, None
        self._has_ts_filter = (
            self._ts_filter_min is not None or self._ts_filter_max is not None
        )

        # Accept either a directory or a single file path.
        if os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        # Out-of-bound statistics:
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # Build the list of Row Groups.
        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)

        # Load schema.json.
        self._load_schema(schema_path, seq_max_lens or {})

        # ---- Pre-compute column index lookup ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # ---- Pre-allocate numpy buffers ----
        B = batch_size
        self._buf_user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        self._buf_item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        self._buf_user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._buf_engineered_dense = np.zeros((B, self.engineered_dense_dim), dtype=np.float32)
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_lens = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        # ---- Pre-compute (col_idx, offset, vocab_size) plans for int columns ----
        self._user_int_plan = []  # [(col_idx, dim, offset, vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            self._user_dense_plan.append((ci, dim, offset))
            offset += dim

        self._item_scalar_dims = sum(1 for _fid, _vs, dim in self._item_int_cols if dim == 1)
        self._item11_offset_len = self.item_int_schema.get_offset_length(11)
        self._item_tail_offsets = {
            fid: self.item_int_schema.get_offset_length(fid)[0]
            for fid in (83, 84, 85)
            if fid in self.item_int_schema.feature_ids
        }

        # Sequence column plan: {domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}")

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """Populate per-group schema information from ``schema_path``."""
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: List[List[int]] = raw['user_int']
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)

        # ---- item_int ----
        self._item_int_cols: List[List[int]] = raw['item_int']
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        # ---- user_dense: [[fid, dim], ...] ----
        self._user_dense_cols: List[List[int]] = raw['user_dense']
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)

        # ---- item_dense (empty) ----
        self.item_dense_schema: FeatureSchema = FeatureSchema()

        # ---- sequence domains ----
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]

            # max_len: from seq_max_lens arg; unspecified domains fall back to 256.
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

        self.engineered_dense_dim = (
            ITEM_STRUCT_STAT_FEATS
            + len(self.seq_domains) * SEQ_TIME_STAT_FEATS_PER_DOMAIN
        )

    def __len__(self) -> int:
        # Ceiling per Row Group; this is an upper bound on the true batch count.
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch(batch)
                # Row-level ts_filter may have eliminated every row in the
                # batch; in that case _convert_batch sets a sentinel and we
                # silently skip without yielding an empty batch.
                if batch_dict.get('__skip_batch__'):
                    continue
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Concatenate the buffered batches, shuffle at the row level, then
        re-slice and yield batch-sized chunks.
        """
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged['label'].shape[0]
        rand_idx = torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- Helpers ----

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """Record out-of-bound indices and (optionally) clip them to 0,
        without printing to the console.
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max'] = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """Dump out-of-bound statistics to a file if ``path`` is provided,
        otherwise to ``logging.info``.
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    @staticmethod
    def _scatter_varlen_2d(
        offsets: "npt.NDArray[np.int64]",
        values: "npt.NDArray[Any]",
        out: "npt.NDArray[Any]",
        max_len: int,
        B: int,
    ) -> "npt.NDArray[np.int64]":
        """Vectorized scatter of an Arrow ``ListArray`` into a pre-allocated
        ``[B, max_len]`` buffer.

        Equivalent to::

            for i in range(B):
                start, end = offsets[i], offsets[i + 1]
                use_len = min(end - start, max_len)
                if use_len > 0:
                    out[i, :use_len] = values[start:start + use_len]

        but executed entirely in numpy without a Python loop. ``out`` must be
        pre-zeroed by the caller; this function only writes the valid region.

        Args:
            offsets: Arrow list offsets, length ``B + 1``.
            values: Flat Arrow values buffer.
            out: Pre-allocated output of shape ``[B, max_len]``; written in
                place. Tail positions ``out[i, lengths[i]:]`` are left as-is.
            max_len: Truncation length per row.
            B: Number of rows in the batch.

        Returns:
            ``lengths`` of shape ``[B]`` with the post-truncation valid length
            for every row (``int64``).
        """
        # raw lengths per row, truncated to max_len
        raw_lens = (offsets[1:] - offsets[:-1]).astype(np.int64, copy=False)
        lengths = np.minimum(raw_lens, max_len)

        total = int(lengths.sum())
        if total == 0:
            return lengths

        # row index for every flat-output element
        row_idx = np.repeat(np.arange(B, dtype=np.int64), lengths)
        # column index = position within each row, i.e. 0..lengths[i]-1
        # cumulative offset of each row's start in the flat layout
        cum = np.empty(B, dtype=np.int64)
        cum[0] = 0
        if B > 1:
            np.cumsum(lengths[:-1], out=cum[1:])
        col_idx = np.arange(total, dtype=np.int64) - cum[row_idx]
        # source index in the flat values buffer
        src_idx = offsets[:-1].astype(np.int64, copy=False)[row_idx] + col_idx

        out[row_idx, col_idx] = values[src_idx]
        return lengths

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """Pad an Arrow ``ListArray`` of ints to shape ``[B, max_len]``.

        Values <= 0 are mapped to 0 (padding). Note: the raw data contains -1
        (missing); currently treated the same way as 0 (padding).

        Returns:
            A tuple ``(padded, lengths)`` where ``padded`` has shape
            ``[B, max_len]`` and ``lengths`` has shape ``[B]``.
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = self._scatter_varlen_2d(offsets, values, padded, max_len, B)

        padded[padded <= 0] = 0
        return padded, lengths

    # Backwards-compatible alias kept for bench_raw_dataset.py and other
    # external callers that pre-date the rename. New code should call
    # `_pad_varlen_int_column` directly.
    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """Pad an Arrow ``ListArray<float>`` to shape ``[B, max_dim]``."""
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_dim), dtype=np.float32)
        self._scatter_varlen_2d(offsets, values, padded, max_dim, B)
        return padded

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors."""
        # ============================================================
        # Row-level ts_filter: applied at the very top so that the entire
        # downstream numpy/scatter pipeline only sees rows that survive
        # the train/valid time split. pa.RecordBatch.filter is a
        # zero-copy gather implemented in C++, far cheaper than running
        # the full _convert_batch on the original B=batch_size rows and
        # then doing torch.index_select on every result tensor at the
        # end.
        #
        # Rows with non-positive timestamps are also dropped from any
        # filtered subset because they have no defensible side of the
        # split.
        # ============================================================
        if self._has_ts_filter:
            ts_np = batch.column(self._col_idx['timestamp']).to_numpy()
            mask = ts_np > 0
            if self._ts_filter_min is not None:
                mask &= ts_np >= self._ts_filter_min
            if self._ts_filter_max is not None:
                mask &= ts_np < self._ts_filter_max
            if not mask.any():
                # Signal the caller to skip this batch entirely; we
                # cannot yield a 0-row tensor (downstream collation
                # assumes positive batch size).
                return {'__skip_batch__': True}
            if not mask.all():
                # Only pay the gather cost when some rows are actually
                # dropped; an all-True mask leaves the batch unchanged.
                batch = batch.filter(pa.array(mask))

        B = batch.num_rows

        # ---- meta ----
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()

        # ---- user_int: write into pre-allocated buffer ----
        # Note: null -> 0 (via fill_null), -1 -> 0 (via arr<=0); missing values
        # are treated the same as padding. Features with vs==0 have no vocab
        # information and are forced to 0 on the dataset side so that the
        # model's 1-slot Embedding (created for vs=0) is never indexed out of
        # range.
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('user_int', ci, arr, vs)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('user_int', ci, padded, vs)
                else:
                    padded[:] = 0
                user_int[:, offset:offset + dim] = padded

        # ---- item_int ----
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('item_int', ci, arr, vs)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('item_int', ci, padded, vs)
                else:
                    padded[:] = 0
                item_int[:, offset:offset + dim] = padded

        # ---- user_dense ----
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            user_dense[:, offset:offset + dim] = padded

        engineered_dense = self._buf_engineered_dense[:B]
        engineered_dense[:] = 0.0
        self._build_item_stat_feats(item_int, engineered_dense)

        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),
            'user_dense_feats': torch.from_numpy(user_dense.copy()),
            'item_int_feats': torch.from_numpy(item_int.copy()),
            'item_dense_feats': torch.zeros(B, 0, dtype=torch.float32),
            'label': torch.from_numpy(labels),
            'timestamp': torch.from_numpy(timestamps),
            'user_id': user_ids,
            '_seq_domains': self.seq_domains,
        }

        # ---- Sequence features: fused padding directly into the 3D buffer ----
        for domain_idx, domain in enumerate(self.seq_domains):
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            # Write directly into the pre-allocated 3D buffer.
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # Fused path: vectorized scatter for every side-info column in a
            # single pass; the per-column lengths are reduced via element-wise
            # max into the shared `lengths` buffer.
            for c, (ci, _slot, _vs) in enumerate(side_plan):
                col = batch.column(ci)
                offs = col.offsets.to_numpy()
                vals = col.values.to_numpy()
                col_lens = self._scatter_varlen_2d(
                    offs, vals, out[:, c, :], max_len, B)
                # lengths[i] := max over columns of per-column length
                np.maximum(lengths, col_lens, out=lengths)

            # Values <= 0 -> 0.
            out[out <= 0] = 0

            # Check out-of-bound values per feature's vocab_size.
            # vs==0 means no vocab info; force the whole slice to 0 so that
            # the model's 1-slot Embedding is never indexed out of range.
            for c, (ci, _slot, vs) in enumerate(side_plan):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            # Time bucketing.
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                # Pad timestamps into shape (B, max_len) via vectorized scatter.
                ts_padded = np.zeros((B, max_len), dtype=np.int64)
                self._scatter_varlen_2d(
                    ts_offs, ts_vals, ts_padded, max_len, B)

                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)
                # np.searchsorted returns values in [0, len(BUCKET_BOUNDARIES)].
                # After +1 the nominal range is [1, len(BUCKET_BOUNDARIES)+1];
                # the upper bound only appears when time_diff exceeds the
                # largest boundary (~1 year) and would index past
                # nn.Embedding(NUM_TIME_BUCKETS=len(BUCKET_BOUNDARIES)+1).
                # Clip raw result to [0, len(BUCKET_BOUNDARIES)-1] so the final
                # bucket id (after +1) stays within [1, len(BUCKET_BOUNDARIES)]
                # and is always a valid Embedding index. Time-diffs beyond the
                # largest boundary collapse into the last bucket.
                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0, len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets
                self._write_seq_time_stat_feats(
                    engineered_dense=engineered_dense,
                    domain_idx=domain_idx,
                    timestamps=timestamps,
                    seq_timestamps=ts_padded,
                    seq_lengths=lengths,
                    max_len=max_len,
                )
            else:
                self._write_seq_time_stat_feats(
                    engineered_dense=engineered_dense,
                    domain_idx=domain_idx,
                    timestamps=timestamps,
                    seq_timestamps=None,
                    seq_lengths=lengths,
                    max_len=max_len,
                )

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())

        result['engineered_dense_feats'] = torch.from_numpy(engineered_dense.copy())
        return result

    def _build_item_stat_feats(
        self,
        item_int: "npt.NDArray[np.int64]",
        engineered_dense: "npt.NDArray[np.float32]",
    ) -> None:
        """Write target-item structural stats into the engineered dense buffer."""
        total_item_dim = max(1, self.item_int_schema.total_dim)
        item_nonzero = (item_int > 0).sum(axis=1).astype(np.float32, copy=False)
        engineered_dense[:, 0] = item_nonzero / float(total_item_dim)

        scalar_nonzero = np.zeros(item_int.shape[0], dtype=np.float32)
        for fid, _vs, dim in self._item_int_cols:
            if dim != 1:
                continue
            offset, _ = self.item_int_schema.get_offset_length(fid)
            scalar_nonzero += (item_int[:, offset] > 0).astype(np.float32, copy=False)
        engineered_dense[:, 1] = scalar_nonzero / float(max(1, self._item_scalar_dims))

        item11_offset, item11_len = self._item11_offset_len
        item11_vals = item_int[:, item11_offset:item11_offset + item11_len]
        item11_nonzero = (item11_vals > 0).sum(axis=1).astype(np.float32, copy=False)
        engineered_dense[:, 2] = item11_nonzero / float(max(1, item11_len))

        tail_count = np.zeros(item_int.shape[0], dtype=np.float32)
        for out_idx, fid in enumerate((83, 84, 85), start=3):
            offset = self._item_tail_offsets.get(fid)
            if offset is None:
                continue
            has_feat = (item_int[:, offset] > 0).astype(np.float32, copy=False)
            engineered_dense[:, out_idx] = has_feat
            tail_count += has_feat
        engineered_dense[:, 6] = tail_count / 3.0

    def _write_seq_time_stat_feats(
        self,
        engineered_dense: "npt.NDArray[np.float32]",
        domain_idx: int,
        timestamps: "npt.NDArray[np.int64]",
        seq_timestamps: Optional["npt.NDArray[np.int64]"],
        seq_lengths: "npt.NDArray[np.int64]",
        max_len: int,
    ) -> None:
        """Write per-domain recency and density features."""
        base = ITEM_STRUCT_STAT_FEATS + domain_idx * SEQ_TIME_STAT_FEATS_PER_DOMAIN
        valid_len = np.minimum(seq_lengths, max_len).astype(np.float32, copy=False)
        engineered_dense[:, base] = valid_len / float(max(1, max_len))

        if seq_timestamps is None:
            return

        valid_mask = seq_timestamps > 0
        valid_count = np.maximum(valid_mask.sum(axis=1), 1).astype(np.float32, copy=False)
        gaps = np.maximum(timestamps.reshape(-1, 1) - seq_timestamps, 0)
        gap_scale = np.log1p(float(BUCKET_BOUNDARIES[-1]))

        min_source = np.where(valid_mask, gaps, np.iinfo(np.int64).max)
        min_gap = min_source.min(axis=1)
        min_gap[min_gap == np.iinfo(np.int64).max] = 0
        max_gap = np.where(valid_mask, gaps, 0).max(axis=1)
        mean_gap = np.where(valid_mask, gaps, 0).sum(axis=1) / valid_count

        engineered_dense[:, base + 1] = np.log1p(min_gap).astype(np.float32) / gap_scale
        engineered_dense[:, base + 2] = np.log1p(mean_gap).astype(np.float32) / gap_scale
        engineered_dense[:, base + 3] = np.log1p(max_gap).astype(np.float32) / gap_scale
        engineered_dense[:, base + 4] = (
            ((gaps <= 86400) & valid_mask).sum(axis=1).astype(np.float32) / valid_count
        )
        engineered_dense[:, base + 5] = (
            ((gaps <= 604800) & valid_mask).sum(axis=1).astype(np.float32) / valid_count
        )


def _collect_rg_with_time(
    pq_files: List[str],
    timestamp_col: str = 'timestamp',
) -> Tuple[List[Tuple[str, int, int, Optional[int], Optional[int]]], int, int]:
    """Collect ``(file, rg_idx, num_rows, min_ts, max_ts)`` for every Row Group.

    Both ``min_ts`` and ``max_ts`` are sourced in this order:
      1. Parquet column statistics (``row_group(i).column(j).statistics``);
         O(1) per RG, no payload read.
      2. Fallback: read the entire ``timestamp`` column for the RG and
         compute ``min`` / ``max`` directly. Slower but only triggered when
         the writer omitted statistics.
      3. ``None`` if the file does not contain ``timestamp_col`` at all.

    Both bounds are needed by the time-aware split: sorting by ``max_ts``
    (the strictest upper-bound per RG) plus dropping any later RG whose
    ``min_ts`` falls inside an earlier RG's range guarantees that every
    validation timestamp is strictly greater than every training timestamp,
    even when individual Row Groups span many days.

    Returns:
        A tuple ``(rg_info, num_with_ts, num_without_ts)``.
    """
    rg_info: List[Tuple[str, int, int, Optional[int], Optional[int]]] = []
    num_with_ts = 0
    num_without_ts = 0
    for f in pq_files:
        pf = pq.ParquetFile(f)
        schema_names = pf.schema_arrow.names
        ts_col_idx: Optional[int] = (
            schema_names.index(timestamp_col)
            if timestamp_col in schema_names else None)
        for i in range(pf.metadata.num_row_groups):
            n_rows = pf.metadata.row_group(i).num_rows
            min_ts: Optional[int] = None
            max_ts: Optional[int] = None
            if ts_col_idx is not None:
                col_meta = pf.metadata.row_group(i).column(ts_col_idx)
                stats = col_meta.statistics
                if stats is not None and stats.has_min_max:
                    try:
                        min_ts = int(stats.min)
                        max_ts = int(stats.max)
                    except (TypeError, ValueError):
                        min_ts = None
                        max_ts = None
                if (min_ts is None or max_ts is None) and n_rows > 0:
                    # Statistics absent or partial: read full ts column once.
                    try:
                        tbl = pf.read_row_group(i, columns=[timestamp_col])
                        if tbl.num_rows > 0:
                            arr = tbl.column(0).to_numpy(zero_copy_only=False)
                            min_ts = int(arr.min())
                            max_ts = int(arr.max())
                    except Exception as exc:  # pragma: no cover - best-effort
                        logging.warning(
                            f"Failed to read timestamp from {f} rg={i}: {exc}")
                        min_ts = None
                        max_ts = None
            if min_ts is None or max_ts is None:
                num_without_ts += 1
            else:
                num_with_ts += 1
            rg_info.append((f, i, n_rows, min_ts, max_ts))
    return rg_info, num_with_ts, num_without_ts


def scan_exact_split_ts(
    data_dir: str,
    valid_ratios: Optional[List[float]] = None,
    timestamp_col: str = 'timestamp',
) -> Dict[str, Any]:
    """One-shot precise scan over every Parquet's ``timestamp`` column to
    pick a row-level ``split_ts`` for train/valid splitting.

    Why this exists
    ---------------
    The Row-Group-statistics-based estimator
    (``decide_row_level_split_threshold``) assumes uniform time density
    inside each Row Group. On heavily skewed data (e.g. 80%+ rows on the
    last day) this severely over-shoots the cut and produces a near-empty
    validation set. This function side-steps the assumption by reading
    the timestamp column itself (one column, ~8 B per row) and computing
    exact quantiles.

    Cost
    ----
    Reads exactly one int64 column per file (zero-copy where possible)
    and sorts the result in-place. For ~1M rows: ~8 MB peak memory and
    ~5–30 s wall-clock depending on disk speed. Called exactly once at
    training startup.

    Args:
        data_dir: Directory containing ``*.parquet`` files (or a single
            file path).
        valid_ratios: List of target valid-side row fractions to evaluate
            in the same scan. Defaults to ``[0.05, 0.10, 0.15]``. Each
            ratio yields one ``split_ts`` recommendation in the result.
        timestamp_col: Column name of the wall-clock timestamp.

    Returns:
        Dict with keys:
          - ``total_rows``      : int, count of rows with ts > 0
          - ``ts_min`` / ``ts_max`` : int (UTC seconds)
          - ``percentiles``     : {p10, p25, p50, p75, p90, p95, p99} -> ts
          - ``splits``          : list of dicts (one per requested ratio):
              {target_valid_ratio, split_ts, train_rows, valid_rows,
               achieved_valid_ratio}
    """
    if valid_ratios is None:
        valid_ratios = [0.05, 0.10, 0.15]
    for r in valid_ratios:
        if not (0.0 < r < 1.0):
            raise ValueError(f"valid_ratio={r!r} not in (0, 1)")

    # ---- Locate parquet files ----
    if os.path.isdir(data_dir):
        import glob as _glob
        pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))
    else:
        pq_files = [data_dir]
    if not pq_files:
        raise FileNotFoundError(f"No .parquet files in {data_dir!r}")

    # ---- Stream-read the timestamp column ----
    import time as _time
    chunks: List[np.ndarray] = []
    t0 = _time.time()
    n_files = len(pq_files)
    logging.info(
        "Exact split scan: reading '%s' column from %d Parquet file(s)...",
        timestamp_col, n_files,
    )
    for idx, f in enumerate(pq_files, 1):
        pf = pq.ParquetFile(f)
        try:
            tbl = pf.read(columns=[timestamp_col])
        except KeyError as exc:
            raise KeyError(
                f"File {f} missing '{timestamp_col}' column. "
                f"Available: {pf.schema_arrow.names}"
            ) from exc
        arr = (
            tbl.column(0)
            .to_numpy(zero_copy_only=False)
            .astype(np.int64, copy=False)
        )
        # Drop non-positive sentinels (treated as "unknown" elsewhere).
        arr = arr[arr > 0]
        chunks.append(arr)
        if idx % 100 == 0 or idx == n_files:
            logging.info(
                "  scanned %d/%d files (%.1fs elapsed, %s rows so far)",
                idx, n_files, _time.time() - t0,
                f"{sum(c.shape[0] for c in chunks):,}",
            )
    if not chunks:
        raise ValueError("No usable timestamps found across input files.")
    sorted_ts = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    del chunks
    sorted_ts.sort()
    n = int(sorted_ts.shape[0])
    logging.info(
        "Exact split scan: sorted %s timestamps in %.1fs total.",
        f"{n:,}", _time.time() - t0,
    )

    # ---- Coarse percentiles ----
    pcts = (10, 25, 50, 75, 90, 95, 99)
    percentiles = {}
    for p in pcts:
        idx = max(0, min(n - 1, int(round(p / 100.0 * (n - 1)))))
        percentiles[f'p{p}'] = int(sorted_ts[idx])

    # ---- Per-ratio split decision ----
    splits = []
    for ratio in valid_ratios:
        target_valid = int(round(ratio * n))
        target_valid = max(1, min(n - 1, target_valid))
        cut_idx = n - target_valid
        split_ts = int(sorted_ts[cut_idx])
        # Recount precisely (handles tied timestamps).
        first_ge = int(np.searchsorted(sorted_ts, split_ts, side='left'))
        valid_count = n - first_ge
        train_count = first_ge
        splits.append({
            'target_valid_ratio': float(ratio),
            'split_ts': split_ts,
            'train_rows': train_count,
            'valid_rows': valid_count,
            'achieved_valid_ratio': valid_count / n,
        })

    return {
        'total_rows': n,
        'ts_min': int(sorted_ts[0]),
        'ts_max': int(sorted_ts[-1]),
        'percentiles': percentiles,
        'splits': splits,
    }


def decide_row_level_split_threshold(
    rg_info_with_ts: List[Tuple[str, int, int, Optional[int], Optional[int]]],
    valid_ratio: float,
    quantile_resolution: int = 200,
) -> Tuple[Optional[int], Dict[str, Any]]:
    """Decide a single ``split_ts`` so that ``rows with ts >= split_ts``
    account for approximately ``valid_ratio`` of all rows.

    The function only consumes per-Row-Group ``(min_ts, max_ts, n_rows)``
    metadata — never row-level data — so it is essentially free even on
    very large datasets.

    Algorithm:
      1. Build a sorted list of candidate split timestamps from every RG's
         min_ts and max_ts (deduped).
      2. For each candidate ``t``, estimate how many rows would land in
         the ``ts >= t`` partition by **uniform-density approximation**
         within each RG: a RG with ``[min_ts, max_ts]`` and ``n_rows``
         contributes ``n_rows * max(0, min(1, (max_ts - t) / (max_ts - min_ts)))``
         rows to the ``>= t`` side. (Constant-time per RG; constant-time
         total when summed across all RGs for one candidate.)
      3. Binary-search the candidate list for the smallest ``t`` such that
         the estimated valid-side row count is ``<= valid_ratio * total``.
         Returning the **smallest** ``t`` gives valid as close as possible
         to ``valid_ratio`` from below — easier on downstream sample
         budgeting than overshooting.

    Args:
        rg_info_with_ts: 5-tuple list ``(file, rg_idx, n_rows, min_ts, max_ts)``
            as produced by ``_collect_rg_with_time``.
        valid_ratio: Target fraction of rows that should land on the valid
            side (``ts >= split_ts``). Typical 0.05 - 0.15.
        quantile_resolution: Soft cap on the number of distinct candidate
            timestamps considered. Set to a positive int to subsample for
            speed when there are millions of unique RG bounds (rare). Each
            quantile is O(n_rgs) to evaluate, so 200 candidates × 1000 RGs
            = 200k ops, sub-second.

    Returns:
        ``(split_ts, info_dict)`` where ``split_ts`` is None if the input
        contains no usable timestamps, and ``info_dict`` carries diagnostic
        fields for logging (estimated train/valid rows, achieved ratio,
        candidate count, time-range string).
    """
    # Filter to RGs that actually have both bounds set.
    usable = [
        (n, mn, mx) for (_f, _i, n, mn, mx) in rg_info_with_ts
        if mn is not None and mx is not None and n > 0
    ]
    if not usable:
        return None, {
            'reason': 'no usable timestamps',
            'total_rows': 0,
            'train_rows_est': 0,
            'valid_rows_est': 0,
        }

    total_rows = sum(n for n, _, _ in usable)
    target_valid_rows = total_rows * float(valid_ratio)

    def _valid_rows_at(t: int) -> float:
        """Estimated count of rows with ts >= t (uniform within each RG)."""
        s = 0.0
        for n, mn, mx in usable:
            if mx < t:           # whole RG strictly before t -> 0 rows
                continue
            if mn >= t:          # whole RG at-or-after t -> n rows
                s += n
                continue
            # Partial overlap: assume rows are uniformly distributed in time.
            span = mx - mn
            if span <= 0:
                # Degenerate (single timestamp == mn == mx).
                # If t <= mn, everything is on the >= side.
                s += n if t <= mn else 0
            else:
                frac = (mx - t) / span
                # Clamp to [0, 1] (paranoia; mathematically guaranteed here).
                if frac < 0.0:
                    frac = 0.0
                elif frac > 1.0:
                    frac = 1.0
                s += n * frac
        return s

    # Build a deduped, sorted candidate list from every RG bound.
    cand_set = set()
    for n, mn, mx in usable:
        cand_set.add(int(mn))
        cand_set.add(int(mx))
    candidates = sorted(cand_set)

    # Optional subsample for very large candidate lists.
    if quantile_resolution > 0 and len(candidates) > quantile_resolution:
        step = max(1, len(candidates) // quantile_resolution)
        candidates = candidates[::step]
        # Always keep the very last bound so we can hit "valid is empty".
        if candidates[-1] != sorted(cand_set)[-1]:
            candidates.append(sorted(cand_set)[-1])

    # Binary search for the smallest t with estimated valid_rows <= target.
    # Note: _valid_rows_at is monotonically non-increasing in t, so this
    # admits a clean bisect.
    lo, hi = 0, len(candidates) - 1
    best_idx = hi  # fallback: largest t (valid empty)
    while lo <= hi:
        mid = (lo + hi) // 2
        est = _valid_rows_at(candidates[mid])
        if est <= target_valid_rows:
            best_idx = mid
            hi = mid - 1   # try a smaller t (more valid rows)
        else:
            lo = mid + 1   # need bigger t (fewer valid rows)

    split_ts = int(candidates[best_idx])
    valid_rows_est = _valid_rows_at(split_ts)
    train_rows_est = total_rows - valid_rows_est
    achieved_ratio = (valid_rows_est / total_rows) if total_rows > 0 else 0.0

    info: Dict[str, Any] = {
        'split_ts': split_ts,
        'total_rows': total_rows,
        'train_rows_est': int(round(train_rows_est)),
        'valid_rows_est': int(round(valid_rows_est)),
        'target_valid_ratio': float(valid_ratio),
        'achieved_valid_ratio': achieved_ratio,
        'num_candidates_evaluated': len(candidates),
        'ts_min': min(int(mn) for _, mn, _ in usable),
        'ts_max': max(int(mx) for _, _, mx in usable),
    }
    return split_ts, info


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    prefetch_factor: int = 4,
    time_aware_split: bool = True,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    Splitting strategy (with ``time_aware_split=True``, the default):

    1. Every Parquet Row Group is annotated with its earliest ``timestamp``
       (read from Parquet column statistics; or, if those are missing, from
       the first row of the RG).
    2. All Row Groups are sorted in ascending time order.
    3. The validation set is taken as the **most recent** ``valid_ratio``
       fraction of RGs; the training set is taken from the **earlier**
       ``train_ratio`` fraction of the remainder.

    This guarantees that every validation timestamp is no earlier than every
    training timestamp, eliminating the temporal leakage that the previous
    ``sorted(glob(...))`` strategy could introduce when filename order did
    not match wall-clock order.

    When ``time_aware_split=False`` (or when the data has no usable
    ``timestamp`` column at all), we fall back to the legacy
    filename-lexicographic ordering and emit a warning so the user knows
    the split may leak.

    Returns:
        A tuple ``(train_loader, valid_loader, train_dataset)``. The third
        element is returned so the caller can access the feature schema
        (``user_int_schema``, ``item_int_schema``, ...) needed to construct
        the model.
    """
    random.seed(seed)

    import glob as _glob
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))

    # ---- Collect Row Groups, with per-RG min timestamp when available ----
    rg_info_with_ts, num_with_ts, num_without_ts = _collect_rg_with_time(
        pq_files, timestamp_col='timestamp')

    # Decide whether to use time-aware ordering.
    # Requirements: feature flag on AND every RG has a min_ts. If any RG
    # is missing a timestamp, we cannot honor the no-leakage promise, so
    # we fall back to legacy ordering with a loud warning.
    use_time_order = time_aware_split and num_without_ts == 0 and num_with_ts > 0
    if time_aware_split and not use_time_order:
        logging.warning(
            "time_aware_split requested but %d/%d Row Groups have no usable "
            "timestamp; falling back to filename-lexicographic split. The "
            "resulting train/valid split MAY leak across time.",
            num_without_ts, num_with_ts + num_without_ts,
        )

    total_rgs = len(rg_info_with_ts)

    # =====================================================================
    # Row-level time split decision (replaces the old RG-level slicing).
    #
    # The old logic could only carve at Row-Group boundaries, which dropped
    # a large fraction of valid rows when a single early RG spanned deep
    # into the test-time window. We now compute one global ``split_ts``
    # from per-RG (min_ts, max_ts, n_rows) statistics; both the train
    # Dataset and the valid Dataset see *all* Row Groups but apply
    # complementary row-level filters in `_convert_batch`. Zero data is
    # silently dropped — every row goes to exactly one of train / valid.
    # Setting ``time_aware_split=False`` falls back to the legacy
    # filename-lexicographic split (no row filter).
    # =====================================================================

    split_ts: Optional[int] = None
    train_ts_filter: Optional[Tuple[Optional[int], Optional[int]]] = None
    valid_ts_filter: Optional[Tuple[Optional[int], Optional[int]]] = None
    train_rows_exact: Optional[int] = None
    valid_rows_exact: Optional[int] = None
    total_rows_exact: Optional[int] = None
    scan_report: Dict[str, Any] = {}

    if use_time_order:
        # ---- Pre-computed split point (preferred fast path) ----
        # We ran ``scan_exact_split_ts`` once on the full TAAC2026 training
        # corpus (1,010,000 rows; ts range 2026-03-18 ~ 2026-03-22) and
        # cached the resulting quantile cuts here. Re-using them at
        # startup skips the ~0.7 s column scan and — more importantly —
        # makes the split deterministic across runs / machines.
        #
        # Mapping is keyed on the *requested* ``valid_ratio``; for any
        # ratio not listed (or for a different dataset where the cached
        # quantile would be wrong), we fall back to a fresh full scan.
        # Cached numbers reproduce the precise scan report:
        #   split_ts=1774220924 (2026-03-22T23:08:44+00:00 UTC)
        #     -> train=908,952  valid=101,048   (exact 10.00% / 10.00%)
        #   split_ts=1774221905 (2026-03-22T23:25:05+00:00 UTC)
        #     -> train=959,489  valid= 50,511   (exact  4.95% / 5.00%)
        #   split_ts=1774219938 (2026-03-22T22:52:18+00:00 UTC)
        #     -> train=858,483  valid=151,517   (exact 14.99% /15.00%)
        _CACHED_SPLITS: Dict[float, Dict[str, int]] = {
            0.10: {'split_ts': 1774220924, 'train_rows': 908952,
                   'valid_rows': 101048, 'total_rows': 1010000},
            0.05: {'split_ts': 1774221905, 'train_rows': 959489,
                   'valid_rows':  50511, 'total_rows': 1010000},
            0.15: {'split_ts': 1774219938, 'train_rows': 858483,
                   'valid_rows': 151517, 'total_rows': 1010000},
        }
        cached = _CACHED_SPLITS.get(round(float(valid_ratio), 4))
        # Sanity check: the cache is only valid if total row count from the
        # current Parquet metadata matches what we recorded. If the dataset
        # has changed we silently fall through to a real scan.
        cached_total = cached['total_rows'] if cached else None
        actual_total_rows = sum(n for (_f, _i, n, _mn, _mx) in rg_info_with_ts)
        cache_hit = (
            cached is not None
            and cached_total == actual_total_rows
            and train_ratio == 1.0  # train_ratio<1.0 still needs sub-scan
        )
        if cache_hit:
            split_ts = int(cached['split_ts'])
            train_rows_exact = int(cached['train_rows'])
            valid_rows_exact = int(cached['valid_rows'])
            total_rows_exact = int(cached['total_rows'])
            train_ts_filter = (None, split_ts)
            valid_ts_filter = (split_ts, None)
            scan_report = {
                'total_rows': total_rows_exact,
                'ts_min': None, 'ts_max': None,
                'percentiles': {},
                'splits': [{
                    'target_valid_ratio': float(valid_ratio),
                    'split_ts': split_ts,
                    'train_rows': train_rows_exact,
                    'valid_rows': valid_rows_exact,
                    'achieved_valid_ratio': valid_rows_exact / total_rows_exact,
                }],
                '_source': 'cached',
            }
            logging.info(
                "Split point loaded from in-code cache (skipping full ts scan): "
                "valid_ratio=%.4f -> split_ts=%s  train=%s  valid=%s  total=%s",
                valid_ratio, split_ts,
                f"{train_rows_exact:,}", f"{valid_rows_exact:,}",
                f"{total_rows_exact:,}",
            )
        else:
            # ---- Precise one-shot scan over the timestamp column ----
            # On heavily skewed data (e.g. 80%+ rows on the last day) the
            # RG-statistics estimator severely over-shoots, so we always do
            # a real scan here. Cost: ~8 MB peak / ~5-30 s wall-clock for
            # ~1M rows, charged exactly once at startup.
            if cached is not None and cached_total != actual_total_rows:
                logging.info(
                    "In-code split cache present for valid_ratio=%.4f but "
                    "row count mismatch (cached_total=%d vs actual=%d); "
                    "running a fresh scan to be safe.",
                    valid_ratio, cached_total, actual_total_rows,
                )
            try:
                scan_report = scan_exact_split_ts(
                    data_dir=data_dir,
                    valid_ratios=[valid_ratio, 0.05, 0.10, 0.15],
                    timestamp_col='timestamp',
                )
                picked = scan_report['splits'][0]
                assert picked['target_valid_ratio'] == valid_ratio
                split_ts = int(picked['split_ts'])
                train_rows_exact = int(picked['train_rows'])
                valid_rows_exact = int(picked['valid_rows'])
                total_rows_exact = int(scan_report['total_rows'])
                train_ts_filter = (None, split_ts)
                valid_ts_filter = (split_ts, None)
            except Exception as exc:
                logging.warning(
                    "Exact split scan failed (%s); falling back to "
                    "RG-statistics estimator. Validation set may be "
                    "inaccurate.", exc,
                )
                split_ts, _split_info_fallback = decide_row_level_split_threshold(
                    rg_info_with_ts, valid_ratio=valid_ratio,
                )
                if split_ts is not None:
                    train_ts_filter = (None, split_ts)
                    valid_ts_filter = (split_ts, None)

    # ``train_ratio < 1.0``: keep only the latest fraction of the train
    # pool (rows with ts < split_ts). With the precise scan we can carve
    # this in row-level units too: pick the (1 - train_ratio) percentile
    # *within the train side* as the lower bound.
    train_lower_ts: Optional[int] = None
    if (use_time_order and train_ratio < 1.0 and split_ts is not None
            and total_rows_exact is not None and train_rows_exact is not None):
        # Re-derive from the same scan: we need the percentile within the
        # train-only sorted timestamps. We do not keep the sorted array
        # around (memory), so re-scan one more time but only on rows
        # < split_ts. This is O(n) memory + sort but rare (only when
        # train_ratio<1.0).
        try:
            sub_report = scan_exact_split_ts(
                data_dir=data_dir,
                valid_ratios=[1.0 - train_ratio],
                timestamp_col='timestamp',
            )
            # Use the percentile as a lower bound over the FULL ts range,
            # then clamp to be < split_ts so we never accidentally extend
            # into valid territory.
            sub_split = sub_report['splits'][0]
            cand_lower = int(sub_split['split_ts'])
            train_lower_ts = min(cand_lower, split_ts)
            train_ts_filter = (train_lower_ts, split_ts)
            logging.info(
                "train_ratio=%.4f -> only rows with ts in [%s, %s) are kept for train.",
                train_ratio, train_lower_ts, split_ts,
            )
            train_rows_exact = int(round(train_rows_exact * train_ratio))
        except Exception as exc:
            logging.warning(
                "Sub-scan for train_ratio failed (%s); ignoring train_ratio.",
                exc,
            )

    # Drop the per-RG (min_ts, max_ts) before downstream code consumes the
    # list: PCVRParquetDataset still wants 3-tuples.
    rg_info = [(f, i, n) for (f, i, n, _mn, _mx) in rg_info_with_ts]

    # Counts for logging. Prefer the exact scan results; fall back to the
    # raw RG row counts if the scan was unavailable.
    if train_rows_exact is not None and valid_rows_exact is not None:
        train_rows_est = train_rows_exact
        valid_rows_est = valid_rows_exact
        total_rows_est = total_rows_exact or (train_rows_exact + valid_rows_exact)
    else:
        train_rows_est = 0
        valid_rows_est = 0
        total_rows_est = sum(r[2] for r in rg_info)

    train_pct = (train_rows_est / total_rows_est * 100.0) if total_rows_est > 0 else 0.0
    valid_pct = (valid_rows_est / total_rows_est * 100.0) if total_rows_est > 0 else 0.0

    # ---------------- SPLIT THRESHOLD REPORT (printed every run) ---------
    logging.info(
        "============== SPLIT THRESHOLD REPORT (row-level, exact scan) =============="
    )
    if use_time_order and split_ts is not None:
        from datetime import datetime, timezone

        def _iso(ts: Any) -> str:
            try:
                return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
            except (OverflowError, OSError, ValueError, TypeError):
                return "<out-of-range>"

        split_ts_iso = _iso(split_ts)
        ts_min = scan_report.get('ts_min')
        ts_max = scan_report.get('ts_max')
        logging.info(
            "Strategy: row-level hard split (precise quantile from full ts column scan)"
        )
        logging.info(
            "Source: %d Parquet files / %d Row Groups; total usable rows=%s",
            len(pq_files), total_rgs, f"{total_rows_est:,}",
        )
        if ts_min is not None and ts_max is not None:
            logging.info(
                "Global timestamp range: ts_min=%s (%s)  ts_max=%s (%s)",
                ts_min, _iso(ts_min), ts_max, _iso(ts_max),
            )
        # Print the full percentile table so you can eyeball the time
        # density without having to re-scan.
        pcts = scan_report.get('percentiles', {})
        if pcts:
            pretty = ", ".join(
                f"{k}={v} ({_iso(v)})" for k, v in pcts.items()
            )
            logging.info("Percentiles (UTC): %s", pretty)
        logging.info(
            "Decision: split_ts=%s (%s UTC)  -> train: ts < split_ts ; valid: ts >= split_ts",
            split_ts, split_ts_iso,
        )
        logging.info(
            "Exact rows: train=%s  valid=%s  total=%s  (train=%.2f%% / valid=%.2f%%; target valid_ratio=%.4f)",
            f"{train_rows_est:,}", f"{valid_rows_est:,}", f"{total_rows_est:,}",
            train_pct, valid_pct, valid_ratio,
        )
        # Also print the alternative-ratio recommendations from the same
        # scan, so you can pick a different valid_ratio next run without
        # paying the scan cost again (just hardcode a different split_ts).
        alt_splits = [s for s in scan_report.get('splits', [])
                      if s['target_valid_ratio'] != valid_ratio]
        for alt in alt_splits:
            logging.info(
                "Alt: valid_ratio=%.4f -> split_ts=%s (%s)  train=%s  valid=%s",
                alt['target_valid_ratio'], alt['split_ts'],
                _iso(alt['split_ts']),
                f"{alt['train_rows']:,}", f"{alt['valid_rows']:,}",
            )
        if train_lower_ts is not None:
            logging.info(
                "train_ratio=%.4f applied -> train further restricted to ts >= %s (%s)",
                train_ratio, train_lower_ts, _iso(train_lower_ts),
            )
        logging.info(
            "Leakage invariant: every train row has ts < split_ts AND every valid row has ts >= split_ts -> OK (no leakage)"
        )
    elif use_time_order:
        logging.warning(
            "Row-level split requested but no split_ts could be derived "
            "(no usable timestamps?). Falling back to no filter; train "
            "and valid will both see all rows -> not safe for evaluation."
        )
    else:
        logging.info(
            "Strategy: filename-lexicographic split (legacy, time_aware_split=False). "
            "No row-level filter applied; both Datasets stream all Row Groups."
        )
    logging.info(
        "==========================================================================="
    )

    # When the legacy lexicographic strategy is selected we still need to
    # split *something*; fall back to RG-level slicing so behavior matches
    # the pre-row-level codebase.
    if not use_time_order or split_ts is None:
        n_valid_rgs = max(1, int(total_rgs * valid_ratio))
        n_train_rgs = total_rgs - n_valid_rgs
        if train_ratio < 1.0:
            n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        train_rg_range = (0, n_train_rgs)
        valid_rg_range = (n_train_rgs, total_rgs)
    else:
        # Row-level split: every Dataset reads every RG; the ts_filter
        # picks the right rows.
        n_train_rgs = total_rgs  # for num_workers capping below
        train_rg_range = None
        valid_rg_range = None

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=train_rg_range,
        clip_vocab=clip_vocab,
        ts_filter=train_ts_filter,
    )

    # Cap num_workers at the number of Row Groups visible to the train
    # Dataset: when there are more workers than RGs, the i % num_workers
    # shard inside __iter__ leaves extra workers permanently empty, which
    # (combined with persistent_workers) silently halves throughput once
    # the busy workers have drained their RGs.
    effective_train_workers = min(num_workers, max(1, len(train_dataset._rg_list)))
    if effective_train_workers != num_workers:
        logging.info(
            f"DataLoader num_workers capped: requested={num_workers}, "
            f"effective={effective_train_workers} (n_train_rgs_visible={len(train_dataset._rg_list)})"
        )

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if effective_train_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = max(1, int(prefetch_factor))

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=effective_train_workers, pin_memory=use_cuda, **_train_kw,
    )

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=valid_rg_range,
        clip_vocab=clip_vocab,
        ts_filter=valid_ts_filter,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0, pin_memory=use_cuda,
    )

    # Final reminder so the row counts are easy to grep even when the
    # threshold report above has scrolled past.
    logging.info(
        "Parquet loaders ready: train_rows~%s, valid_rows~%s, batch_size=%d, "
        "buffer_batches=%d, num_workers=%d, ts_filter(train)=%s, ts_filter(valid)=%s",
        f"{train_rows_est:,}", f"{valid_rows_est:,}",
        batch_size, buffer_batches, effective_train_workers,
        train_ts_filter, valid_ts_filter,
    )

    return train_loader, valid_loader, train_dataset
