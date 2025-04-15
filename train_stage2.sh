#!/bin/bash

WORKDIR=${1:-./exp/stage2}
echo $WORKDIR
mkdir -p $WORKDIR

export PYTHONPATH=`pwd`:$PYTHONPATH
export PROJ=1
export CLIP=1
export V15=1


WANDB_API_KEY=xxxxx WANDB_MODE=$WANDB_MODE CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 \
    aor/train/train.py \
    --model_name_or_path exp/stage1\
    --dataset_config ./aor/configs/test_train.py \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --bf16 True \
    --output_dir $WORKDIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 150 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.003 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --lazy_preprocess True \
    --report_to 'wandb' \
    --deepspeed './aor/configs/deepspeed_stage1.json' \
    --seed 0 \
    --dataloader_num_workers 4 \
    --ddp_timeout 1800000 \
    | tee $WORKDIR/train.log
