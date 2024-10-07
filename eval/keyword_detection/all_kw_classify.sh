export search_dir=$1

if [ ! -d $search_dir/kw_out ]; then
    mkdir $search_dir/kw_out
    echo "Folder ${search_dir}/kw_out created."
else
    echo "Folder ${search_dir}/kw_out already exists."
fi

for entry in `ls $search_dir`; do
    if [ -f "$search_dir/$entry" ]; then
        bash keyword_detection.sh $search_dir/$entry "$search_dir/kw_out/$entry" > "$search_dir/kw_out/${entry/.json/.out}"
    fi
done