# Rex-Omni Integration Notes for TT-Occ

## Scope

This note documents runtime behavior, validation ideas, and fallback logic for the `rexomni` semantic provider used by TT-Occ online test-time precompute.

## Runtime Contract

- Provider name in TT-Occ: `semantic_prefix=rexomni`
- Per-frame outputs expected by main pipeline:
  - `{semantic_prefix}_{cam}/{frame}.png` (semantic id map, uint8)
  - `{semantic_prefix}_{cam}/{frame}_instance.png` (instance id map, uint16)
- Instance IDs are frame-local. Cross-frame identity consistency is not required by current RAFT dynamic-mask logic.

## Dependencies

- `qwen_vl_utils`
- `transformers==4.51.3`
- `accelerate==1.10.1`
- `segment-anything`
- SAM checkpoint: `sam_vit_h_4b8939.pth`
- Rex-Omni model: `IDEA-Research/Rex-Omni`

## FlashAttention

- Preferred: `flash-attn==2.7.4.post1` (faster inference).
- If unavailable, TT-Occ keeps a fallback path to eager attention in `rex_omni/wrapper.py`.

## Quick Sanity Check

1. Run one frame online through TT-Occ precompute path (`SEMANTIC_PREFIX=rexomni`).
2. Confirm files exist for all 6 cameras:
   - `precomputed/<scene>/rexomni_0/00.png`
   - `precomputed/<scene>/rexomni_0/00_instance.png`
3. Confirm no missing-instance-file errors in `scene/dataset_readers.py`.

## Common Issues

- Missing HF weights: download `IDEA-Research/Rex-Omni` first.
- Missing SAM checkpoint: ensure `submodules/Rex-Omni/checkpoints/sam_vit_h_4b8939.pth`.
- Tokenizer/hub mismatch: keep `huggingface_hub==0.36.2` with `transformers==4.51.3`.
