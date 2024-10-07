#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export model=$1
export attack=$2 # attngcg or gcg
export method=$3 # direct or ICA or AutoDAN
export ntrain=$4
export folder="../results_individual_${method}_${attack}_${model}"
export data_offset=$5

# Create results folder if it doesn't exist
if [ ! -d $folder ]; then
    mkdir $folder
    echo "Folder ${folder} created."
else
    echo "Folder ${folder} already exists."
fi

python -u ../main.py \
    --config="../configs/individual_${model}.py" \
    --config.train_data="../../data/advbench/harmful_behaviors_${method}.csv" \
    --config.attack=$attack \
    --config.result_prefix="${folder}/individual_${method}_${model}_offset${data_offset}" \
    --config.n_train_data=$ntrain \
    --config.data_offset=$data_offset \
    --config.test_steps=20 \
    --config.model=$model  \
    --config.test_case_path="../testcase/${model}_${attack}_${method}.json"
