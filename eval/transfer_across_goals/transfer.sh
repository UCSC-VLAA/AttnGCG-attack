export model=$1
export attack=$2 # attngcg or gcg
export method="transfer"

if [ ! -d raw_testcase ]; then
    mkdir raw_testcase
    echo "Folder raw_testcase created."
else
    echo "Folder raw_testcase already exists."
fi

if [ ! -d use_testcase ]; then
    mkdir use_testcase
    echo "Folder use_testcase created."
else
    echo "Folder use_testcase already exists."
fi

cp ../../data/advbench/blank_transfer_test_100.json .
cp ../../data/advbench/blank_transfer_train_25.json .
cp ../../experiments/testcase/${model}_${attack}_${method}.json raw_testcase/

python transfer_suffix.py \
    --model=$model \
    --attack=$attack \
    --method=$method