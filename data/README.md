# Data (gitignored)

Nothing in this folder is committed. Images never go into git.

## Where things live on Colab

| What | Where | Why |
|---|---|---|
| Raw Kaggle downloads | `/content/data/raw` (local Colab disk) | Fast, large, and re-downloadable. Never store on Drive. |
| Processed images (crop + CLAHE + resize) | `Drive/XRetina_Shared/processed/` | Shared by all three members |
| Split manifest | `Drive/XRetina_Shared/splits/manifest.csv` | Single source of truth for train/val/test |
| Checkpoints | `Drive/XRetina_Shared/checkpoints/` | Shared model weights |

Free Drive quota is 15 GB, so the EyePACS subset is kept small (see `configs/datasets.yaml`).

## Kaggle API

1. Kaggle > Settings > Create New Token (downloads `kaggle.json`).
2. In Colab, add Secrets `KAGGLE_USERNAME` and `KAGGLE_KEY` and enable notebook access.
   The bootstrap cell writes `~/.kaggle/kaggle.json` for you. Never commit `kaggle.json`.
3. Download example:

```bash
kaggle datasets download -d mariaherrerot/aptos2019 -p /content/data/raw --unzip
```

## Datasets in this project

| Dataset | Used in | Source |
|---|---|---|
| APTOS 2019 | FYP-I training | `mariaherrerot/aptos2019` |
| EyePACS | FYP-I training | `dreamer07/eyepacs` (mirror unverified) |
| IDRiD | FYP-II only | No link in proposal yet |
| Messidor-2 | FYP-II only | `mariaherrerot/messidor2preprocess` |

If you work locally instead of Colab, place raw data under `data/raw/` and set `XRETINA_RAW_DIR`.
