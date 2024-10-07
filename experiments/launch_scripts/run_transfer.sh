#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export model=$1
export attack=$2
export ntrain=$3
export folder="../results_transfer_goals_${model}"
export data_offset=$4

# Create results folder if it doesn't exist
if [ ! -d $folder ]; then
    mkdir $folder
    echo "Folder ${folder} created."
else
    echo "Folder ${folder} already exists."
fi

python -u ../main.py \
    --config="../configs/transfer_${model}.py" \
    --config.train_data="../../data/advbench/harmful_behaviors_transfer.csv" \
    --config.attack=$attack \
    --config.result_prefix="${folder}/transfer_${model}_offset${data_offset}_progressive${ntrain}" \
    --config.n_train_data=$ntrain \
    --config.data_offset=$data_offset \
    --config.test_steps=1 \
    --config.model=$model  \
    --config.setup=$setup \
    --config.stop_on_success=True \
    --config.progressive_goals=True \
    --config.test_case_path="../testcase/${model}_${attack}_transfer.json"

