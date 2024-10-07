# sample train data (default 100 behaviors for training)
cd data
for method in direct ica autodan transfer
do
    python sample_train_data.py --method=$method   # choices=['direct', 'ica', 'autodan', 'transfer']
done

if [ ! -d ../experiments/testcase ]; then
    mkdir ../experiments/testcase
    echo "Folder ../experiments/testcase created."
else
    echo "Folder ../experiments/testcase already exists."
fi

cd advbench
for file in `ls | grep "blank"` 
do
    cp  $file ../../experiments/testcase
done