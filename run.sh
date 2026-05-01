#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: RankMixer NS tokenizer (no ns_groups.json required) ----
# Speed knobs enabled by default (all are AUC-safe):
#   --use_amp            : bf16 autocast on CUDA (no-op on CPU)
#   --use_tf32           : TF32 matmul on Ampere+
#   --grad_clip_foreach  : faster fused clip_grad_norm_
#   --tqdm_min_interval  : throttle terminal redraws
#   --prefetch_factor 4  : deeper DataLoader queue
# Split policy:
#   --time_aware_split   : Row Groups are sorted by min timestamp; train uses
#                          the earliest part and valid uses the most recent
#                          tail. Eliminates temporal leakage that the legacy
#                          filename-lexicographic split could introduce.
#                          (Use --no_time_aware_split to revert.)
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --rank_mixer_mode ffn_only \
    --loss_type focal \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --seq_max_lens 'seq_a:384,seq_b:256,seq_c:512,seq_d:768' \
    --lr 2e-4 \
    --sparse_lr 0.08 \
    --dropout_rate 0.02 \
    --batch_size 256 \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --use_tf32 \
    --grad_clip_foreach \
    --tqdm_min_interval 0.5 \
    --prefetch_factor 4 \
    --time_aware_split \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
