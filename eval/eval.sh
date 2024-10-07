export model=$1 # ['llama2_chat_7b', 'llama2_chat_13b', 'gemma_7b', 'mistral_7b', "mpt_7b", "mixtral_8x7b", "gemma_2b", "llama3_8b_it"]
export attack=$2 # attngcg or gcg
export method=$3 # direct or ica or autodan or transfer

if [ ! -d testcases ]; then
    mkdir testcases
    echo "Folder testcases created."
else
    echo "Folder testcases already exists."
fi

# generate responses
if [ $method == "transfer" ]; then
    cd transfer_across_goals
    bash transfer.sh $model $attack
    cp use_testcase/${model}_${attack}_transfer_test_100.json ../testcases/
    cp use_testcase/${model}_${attack}_transfer_train_25.json ../testcases/
    cd ../gene_response
    bash eval_hf.sh ${model} ${attack} transfer_test_100
    bash eval_hf.sh ${model} ${attack} transfer_train_25
else
    cp ../experiments/testcase/${model}_${attack}_${method}.json testcases/
    cd gene_response
    bash eval_hf.sh ${model} ${attack} ${method}
fi

# evaluate responses
# use keyword detection method to evaluate attack success rate
cd ../keyword_detection
bash all_kw_classify.sh ../generations

## use GPT-4 to evaluate attack success rate
# cd ../gpt4_judge
# bash all_gpt_classify.sh ../generations