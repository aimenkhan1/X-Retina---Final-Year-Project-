# Notebooks

One notebook per pipeline stage, numbered. Every notebook starts with the bootstrap cell from `00_colab_setup.ipynb`
(copy-paste version: `docs/colab_bootstrap.py`).

| Notebook | Stage | Owner | Calendar week | Status |
|---|---|---|---|---|
| `00_colab_setup.ipynb` | Setup / bootstrap | Aiman | Week 7-8 | Ready |
| `01_data_collection.ipynb` | Stage 1: download APTOS + EyePACS subset | Aiman | Week 8 | Next |
| `02_preprocessing.ipynb` | Stage 2: crop, CLAHE, resize, splits | Aiman | Week 8 | Next |
| `03_dataloaders_and_trainer_check.ipynb` | Dataloaders, trainer smoke test | Maheen | Week 9 | Planned |
| `04_baseline_training.ipynb` | Baseline EfficientNet-B4 training | Amna | Week 10 | Planned |
| `05_imbalance_comparison.ipynb` | Cross-entropy vs Focal Loss vs SMOTE | Aiman | Weeks 9-12 | Planned |
| `06_hybrid_model.ipynb` | Hybrid (Transformer + SE/CBAM) first run | Aiman | Weeks 13-15 | Planned |

Keep notebook outputs small and clear large outputs before committing (see README, "Notebook hygiene").
