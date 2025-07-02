data_root=~/Data/new_extracted_nuscenes_val

cd submodules/vggt
python main.py $data_root 
echo "Depth extraction done"