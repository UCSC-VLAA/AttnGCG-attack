export model=$1
export attack=$2
export method=$3

if [ ! -d "../generations" ]; then
    mkdir "../generations"
    echo "Folder ../generations created."
else
    echo "Folder ../generations already exists."
fi

python -u eval_hf.py \
    --test_case_path="../testcases/${model}_${attack}_${method}.json" \
    --generations_path="../generations/${model}_${attack}_${method}.json" \
    --model=$model