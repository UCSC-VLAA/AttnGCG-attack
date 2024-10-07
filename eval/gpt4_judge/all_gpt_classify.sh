export search_dir=$1

if [ ! -d $search_dir/gpt4_out ]; then
    mkdir $search_dir/gpt4_out
    echo "Folder ${search_dir}/gpt4_out created."
else
    echo "Folder ${search_dir}/gpt4_out already exists."
fi

for entry in `ls $search_dir`; do
    if [ -f "$search_dir/$entry" ]; then
        bash gpt_classify.sh $search_dir/$entry $search_dir/gpt4_out/$entry "$search_dir/gpt4_out/${entry/.json/_no.json}" "$search_dir/gpt4_out/${entry/.json/_yes.json}" > "$search_dir/gpt4_out/${entry/.json/.out}"
    fi
done