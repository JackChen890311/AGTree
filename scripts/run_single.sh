#!bin/bash

GPU_ID=0
SEED=0
NODE="v0" # Format: {chr}{num}, e.g. v0, v1, v2, v3
PROMPT="object object"
PARENT="canada_bear"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python main_singleseed.py \
    --parent_data_dir $PARENT \
    --prompt "${PROMPT}" \
    --node $NODE \
    --test_name $NODE \
    --GPU_ID "${GPU_ID}" \
    --seed $SEED \
    --apply_otsu \
    --random_drop 0.8 \
    --random_drop_start_step 500 \
    --attention_start_step 100 \
    --attention_save_step 50 \
    --fused_res 16 \
    --ema_beta 0.95 \
    --run_validation