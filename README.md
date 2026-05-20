# CICF — Causal Intervention via CounterFactual Gradient

Facial expression recognition with robustness to spurious correlations (color confounders).

## Quick Start

> **Important**: Run all commands from `project_package/`, where `dataset_classified_biased/` is located.

### Requirements

```bash
pip install torch torchvision numpy scikit-learn matplotlib pillow
```

### ERM Training
```bash
python script/train.py \
    --mode erm \
    --epochs 50 \
    --train_batch_size 84 \
    --num_workers 0 \
    --checkpoint_path checkpoints/erm_best.pt \
    --log_name train_erm.txt
```

### CICF Training
```bash
python script/train.py \
    --mode cicf \
    --epochs 50 \
    --train_batch_size 84 \
    --g_plus_batch_size 256 \
    --k_per_class 3 \
    --alpha 0.05 \
    --num_workers 0 \
    --checkpoint_path checkpoints/cicf_best.pt \
    --log_name train_cicf.txt
```

### Visualize
```bash
# Gray counterfactual Grad-CAM (Original vs Gray-CF for ERM and CICF)
python script/visualize_confounder_counterfactual_green_gray.py \
    --erm_checkpoint checkpoints/erm_best.pt \
    --cicf_checkpoint checkpoints/cicf_best.pt \
    --split test --scope all --max_examples 32
```

## Files
- `script/` — All source code
  - `train.py` — Training script (two clear branches: ERM / CICF)
  - `model.py` — ResNet-18 backbone (FeatureExtracter + Classifier)
  - `dataset.py` — DataLoader with color-confounded dataset
  - `cicf_utils.py` — KMeans clustering, G⁺ batch sampling, evaluation
  - `visualize_confounder_counterfactual_green_gray.py` — Gray counterfactual Grad-CAM
- `dataset_config.yaml` — Color confounder configuration
- `dataset_classified_biased/` — Dataset (train / valid / test)
- `logs/` — Training logs (ERM & CICF 20-epoch runs)

See `PROJECT_REPORT.md` for the full methodology, experiment design, and results analysis.
