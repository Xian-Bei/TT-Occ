# MapAnything Integration Notes for TT-Occ

## Scope

This note documents runtime behavior, validation ideas, and compatibility details for the `mapanything` depth provider used by TT-Occ online test-time precompute.

## Runtime Contract

- Provider name in TT-Occ: `depth_prefix=mapanything`
- Per-frame outputs expected by main pipeline:
  - `{depth_prefix}_{cam}/{frame}.npy` (depth map, float32)
- Frame resolution is aligned to TT-Occ camera branch path (`294x518` before downstream use).

## Dependencies

- Editable install of `submodules/map-anything`
- `pycolmap==3.10.0`
- `lightglue` (git install)

## Compatibility Notes

- TT-Occ includes a compatibility patch for HF config where `mlp_layer` may appear as a string placeholder.
- Keep shared versions from root README to avoid environment drift:
  - `numpy==1.26.4`
  - `pillow==10.4.0`
  - `huggingface_hub==0.36.2`

## Quick Sanity Check

1. Run one frame online through TT-Occ precompute path (`DEPTH_PREFIX=mapanything`).
2. Confirm files exist for all 6 cameras:
   - `precomputed/<scene>/mapanything_0/00.npy`
3. Confirm RAFT receives in-memory or on-disk depth without missing-file errors.

## Common Issues

- Model init error around MLP layer: ensure this repo's patched `mapanything` model file is used.
- Out-of-memory: reduce scene batching (online mode already processes per-frame).
