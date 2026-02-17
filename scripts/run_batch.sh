#!bin/bash

GPU_ID=0
SEED=0
NODE="v0" # Format: {chr}{num}, e.g. v0, v1, v2, v3
PROMPT="object object"
IN="input_concepts/"
OUT="outputs/"
EXP_FILE_NAME="${OUT}/exp.txt"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

train () {
    local NODE=$1
    echo "Training node $NODE"
    python main_singleseed.py \
        --parent_data_dir "$PARENT" \
        --prompt "${PROMPT}" \
        --node "$NODE" \
        --test_name "$NODE" \
        --GPU_ID "${GPU_ID}" \
        --seed "$SEED" \
        --apply_otsu \
        --random_drop 0.8 \
        --random_drop_start_step 500 \
        --attention_start_step 100 \
        --attention_save_step 50 \
        --fused_res 16 \
        --ema_beta 0.95 \
        --run_validation
}

start_exp(){
    read -p "Do you want to remove all old outputs? (y/n): " REMOVE
    if [[ "$REMOVE" == "y" ]]; then
        echo "Removing all old outputs"
        rm -rf "${OUT}"
        mkdir -p "${OUT}"

        mkdir -p "$(dirname "$EXP_FILE_NAME")"
        read -p "Enter details of the experiment: " DETAILS
        echo "$DETAILS" > "$EXP_FILE_NAME"
        echo "Experiment details saved to $EXP_FILE_NAME"
        echo "Starting new experiments..."
    else
        echo "Keeping old outputs and continuing"
        cat $EXP_FILE_NAME
        echo "."
        echo "Continuing existing experiments..."
    fi
}


start_exp

for PARENT in $(ls "$IN"); 
do
    echo "Parent: $PARENT"

    # if exist "$OUT/$PARENT" then continue
    if [ -d "${OUT}/${PARENT}" ]; then
        echo "Output directory for $PARENT already exists. Skipping..."
        continue
    fi

    train $NODE
done
