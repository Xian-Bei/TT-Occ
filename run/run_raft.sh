data_root=~/Data/new_extracted_nuscenes_val

cd submodules/RAFT
python main.py $data_root
echo "Optical flow extraction done"