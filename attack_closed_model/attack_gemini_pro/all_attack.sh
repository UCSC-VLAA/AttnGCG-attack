export search_dir=$1

if [ ! -d $search_dir/generation ]; then
    mkdir $search_dir/generation
    echo "Folder ${search_dir}/generation created."
else
    echo "Folder ${search_dir}/generation already exists."
fi

if [ ! -d $search_dir/icl ]; then
    mkdir $search_dir/icl
    echo "Folder ${search_dir}/icl created."
else
    echo "Folder ${search_dir}/icl already exists."
fi

if [ ! -d $search_dir/out ]; then
    mkdir $search_dir/out
    echo "Folder ${search_dir}/out created."
else
    echo "Folder ${search_dir}/out already exists."
fi

for entry in `ls $search_dir/testcase`; do
    if [ -f "$search_dir/testcase/$entry" ]; then
        bash attack_gemini.sh $search_dir/testcase/$entry $search_dir/generation/$entry $search_dir/icl/$entry > "$search_dir/out/${entry/.json/.out}"
    fi
done