export source_gene_path=$1
export output_path=$2
export dest_no_path=$3
export dest_yes_path=$4

python -u gpt_classify.py \
    --source_gene_path=$source_gene_path \
    --output_path=$output_path \
    --dest_no_path=$dest_no_path \
    --dest_yes_path=$dest_yes_path