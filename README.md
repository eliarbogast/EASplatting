# EASplatting

Evolutionary search over post-hoc pruning thresholds for pretrained
[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) scenes:
fewer Gaussians and faster rendering with minimal quality loss, with no retraining.
Rendering uses [gsplat](https://github.com/nerfstudio-project/gsplat) 1.5.3; evaluation
follows the [NerfBaselines](https://nerfbaselines.github.io/mipnerf360) Mip-NeRF 360 protocol.

- **Run it locally:** [`docs/LOCAL_PIPELINE.md`](docs/LOCAL_PIPELINE.md)
- **Background, metrics, fitness design, pitfalls:** [`docs/GS_PRUNING_NOTES.md`](docs/GS_PRUNING_NOTES.md)
- **Current status / next steps:** [`HANDOFF.md`](HANDOFF.md)

```bash
bash scripts/setup_local.sh && conda activate gsprune
bash scripts/get_data.sh
python evaluate.py --scene pretrained/nb/gaussian-splatting/mipnerf360/bonsai/point_cloud.ply \
    --cameras data/360_v2/bonsai --images images_2 --split test --out results/eval_bonsai
python -m pytest -q tests     # CPU tests with a fake renderer
```

Datasets, checkpoints and results are downloaded/generated locally and git-ignored.
