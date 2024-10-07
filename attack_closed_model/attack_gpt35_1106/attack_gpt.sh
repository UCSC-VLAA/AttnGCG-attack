export source_testcase_path=$1
export output_generation_path=$2
export icl_path=$3

python -u attack_gpt.py \
    --source_testcase_path=$source_testcase_path \
    --output_generation_path=$output_generation_path \
    --icl_path=$icl_path