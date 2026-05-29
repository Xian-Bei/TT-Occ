data_root=~/Data/new_extracted_nuscenes_val
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${PROJECT_ROOT}/precomputed"
mkdir -p "$output_root"

cd submodules/VGGT
python main.py "$data_root" "$output_root"
echo "Depth extraction done (saved under ${output_root})"
