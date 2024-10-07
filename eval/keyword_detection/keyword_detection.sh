export source_gene_path=$1
export dest_gene_path=$2

python -u keyword_detection.py \
    --source_gene_path=$source_gene_path \
    --dest_gene_path=$dest_gene_path 