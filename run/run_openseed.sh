data_root=~/Data/new_extracted_nuscenes_val

cd submodules/OpenSeeD
python main.py evaluate --conf_files configs/openseed/openseed_swint_lang.yaml \
    --user_dir $data_root \
    --overrides WEIGHT model_state_dict_swint_51.2ap.pt 
echo "Semantic extraction done"