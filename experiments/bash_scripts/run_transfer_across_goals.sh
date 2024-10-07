export model=$1
export attack=$2
export offset=$3
export folder=../bash_scripts/out_${model}_${attack}_transfer

cd ../testcase

if [ ! -f ${model}_${attack}_transfer.json ]; then
    cp blank_transfer_train_25.json ${model}_${attack}_transfer.json
    echo "File ${model}_${attack}_transfer.json created."
else
    echo "File ${model}_${attack}_transfer.json already exists."
fi

cd ../launch_scripts

if [ ! -d $folder ]; then
    mkdir $folder
    echo "Folder $folder created."
else
    echo "Folder $folder already exists."
fi

bash run_transfer.sh $model $attack 25 $offset > ${folder}/${offset}.out &

wait