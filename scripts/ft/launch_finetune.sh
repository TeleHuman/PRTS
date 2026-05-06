#!/bin/bash

# Save the script path for reproducibility
export LAUNCH_SCRIPT=$(realpath $0)

DEBUG=false
if [ "$DEBUG" = true ]; then
  GPUS=1
  logging_steps=2
  PER_DEVICE_BATCH_SIZE=1
fi



logging_steps=100
PER_DEVICE_BATCH_SIZE=8
GPUS=4
save_steps=10000
# report=none

GPUS=${GPUS:-8}
num_workers=4
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=1
report=${report:-wandb}
save_steps=${save_steps:-10000}
logging_steps=${logging_steps:-100}

# multi-gpu training command initialization
ACCELERATE_ARGS="--num_processes=${GPUS}"

# specify datasets
dataset_config_path=configs/post-train/libero.yaml
dataset_name=$(basename ${dataset_config_path%.*})
embodiment_tag=libero_panda
echo $dataset_name

# unused
vlm_base_path=$HF_HUB_CACHE/models--Qwen--Qwen3-VL-4B-Instruct
rope_type=qwen3_vl
image_factor=32

lr=1e-5
mlr=1e-5
vlr=1e-5
action_lr=1e-4
weight_decay=1e-8

chunk_size=20 # 50
action_dim=32
max_train_steps=30000
dit_model_type=DiT-L

mipx=64
mapx=144
state_mode=QUANTILE
lerobot_dataset_only=True
deepspeed=zero2
training_phase=post-training
gradient_checkpointing=False
ratio_train_fast_tokens=0

gamma=0.995

model_name_or_path=TeleEmbodied/PRTS-4B  # local path or HF repo id
resume_path=
action_tokenizer_path=

max_train_steps_k=$(printf "%.0fK" "$((${max_train_steps}/1000))")
run_name=prts_sft_${training_phase}_lr${lr}_vlr${vlr}_mlr${mlr}_${state_mode}_ck${chunk_size}_bs${PER_DEVICE_BATCH_SIZE}_gpu${GPUS}_s${max_train_steps_k}_warmup5k_${deepspeed}_${dataset_name}_total_steps${max_train_steps_k}

echo "RUN NAME: $run_name"
export DATASET_NUM_PROCESSES=8
export TOKENIZERS_PARALLELISM=false

export WANDB_MODE=offline
export WANDB_DISABLE_SERVICE=true
export WANDB_PROJECT=PRTS_v1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=$GPUS scripts/train.py \
    ${resume_path:+--output-dir $resume_path} \
    ${model_name_or_path:+--model-name-or-path $model_name_or_path} \
    ${action_tokenizer_path:+--action-tokenizer-path $action_tokenizer_path} \
    ${deepspeed:+--deepspeed configs/${deepspeed}.json} \
    --vlm-name-or-path ${vlm_base_path} \
    --rope-type ${rope_type} \
    --data-path ${dataset_config_path} \
    --embodiment-tag ${embodiment_tag} \
    --train-lerobot-only ${lerobot_dataset_only} \
    --training-phase ${training_phase} \
    --state-mode ${state_mode} \
    --chunk-size ${chunk_size} \
    --max-action-dim ${action_dim} \
    --ratio-fast-action-tokens ${ratio_train_fast_tokens} \
    --dataloader-num-workers ${num_workers} \
    --freeze-vision-tower False \
    --freeze-llm False \
    --freeze-merger False \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --max-steps ${max_train_steps} \
    --per-device-train-batch-size ${PER_DEVICE_BATCH_SIZE} \
    --gradient-accumulation-steps ${GRADIENT_ACC} \
    --image-min-pixels $((${mipx} * ${image_factor} * ${image_factor})) \
    --image-max-pixels $((${mapx} * ${image_factor} * ${image_factor})) \
    --video-min-pixels $((${mipx} * ${image_factor} * ${image_factor})) \
    --video-max-pixels $((${mapx} * ${image_factor} * ${image_factor})) \
    --fps 2.0 \
    --learning-rate ${lr} \
    --merger-lr ${mlr} \
    --vision-lr ${vlr} \
    --action-lr ${action_lr} \
    --weight-decay ${weight_decay} \
    --optim adamw_torch_fused \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1 \
    --warmup-steps 5000 \
    --lr-scheduler-type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr":1e-6}' \
    --logging-steps ${logging_steps} \
    --save-strategy steps \
    --save-steps ${save_steps} \
    --save-total-limit 3 \
    --report-to ${report} \
    --run-name ${run_name} \
    --ignore_data_skip False \
    --attn-implementation flash_attention_3 \
    --seed 42 \
    --gamma ${gamma} \
    --use_mot_action_expert True