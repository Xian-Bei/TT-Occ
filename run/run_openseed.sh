data_root=~/Data/new_extracted_nuscenes_val
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${PROJECT_ROOT}/precomputed"
mkdir -p "$output_root"

cd submodules/OpenSeeD
python main.py evaluate --conf_files configs/openseed/openseed_swint_lang.yaml \
    --user_dir "$data_root" \
    --output_dir "$output_root" \
    --overrides WEIGHT model_state_dict_swint_51.2ap.pt
echo "Semantic extraction done (saved under ${output_root})"
