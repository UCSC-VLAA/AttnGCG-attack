export base_model=$1 # choices=['llama2_chat_7b', 'llama2_chat_13b', 'gemma_7b', 'mistral_7b', "mpt_7b", "mixtral_8x7b", "gemma_2b", "llama3_8b_it"]
export base_attack=$2 # choices=['attngcg', 'gcg']
export target_model=$3 # choices=['gemini_pro', 'gpt4_1106', 'gpt35_0125', 'gpt35_0613', 'gpt35_1106', 'gpt35_instruct']

cd attack_${target_model}
if [ ! -d testcase ]; then
    mkdir testcase
    echo "Folder testcase created."
else
    echo "Folder testcase already exists."
fi
cp ../../experiments/testcase/${base_model}_${base_attack}_direct.json testcase/
bash all_attack.sh .