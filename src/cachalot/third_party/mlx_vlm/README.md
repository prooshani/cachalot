# Vendored from mlx-vlm

These files are copied unmodified from [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) at commit
`ad4a3cc` (2026-09-24), MIT licence (see `LICENSE`). They are the smallest set that imports
`models/glm5_next/language.py`, the GLM-5.3-Flash (`glm5_next`) language model, without mlx-vlm's
package initialiser (generation, conversion, processors, audio). The package `__init__.py` files
here are empty on purpose.

Cachalot uses the GLM model code as it is and replaces only each MoE layer's `switch_mlp` with its
own SSD expert streaming (`cachalot.glm`). To update, re-copy the same files from a newer commit and
re-run `tests/test_glm_backend.py`.
