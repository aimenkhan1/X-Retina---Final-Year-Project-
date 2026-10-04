# X-Retina

**An Explainable Deep Learning Framework for Diabetic Retinopathy Grading and Lesion Localization**
FAST-NUCES Karachi, FYP Fall 2026. Team: Aiman Nadeem Khan, Amna Rais, Maheen Sodha.

[![Open 00 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/x-retina/blob/main/notebooks/00_colab_setup.ipynb)

## Scope

**FYP-I (this repo, now):** literature validation, dataset acquisition and preprocessing (CLAHE), class-imbalance methods (Focal Loss, SMOTE), baseline EfficientNet-B4 training, and the initial EfficientNet-B4 + Transformer + SE/CBAM hybrid.
**FYP-II (placeholders only):** ablation, XAI (Grad-CAM, Grad-CAM++, Score-CAM, Saliency) with Dice vs IDRiD, benchmarking and Messidor-2, dashboard. Folders `src/xai`, `src/ablation`, `src/benchmark`, `dashboard/` are empty on purpose.

## Structure

```
x-retina/
├── notebooks/     Colab-ready .ipynb, one per pipeline stage
├── src/           reusable modules (data, preprocessing, losses, imbalance, models, training, evaluation)
│   ├── xai/ ablation/ benchmark/   FYP-II placeholders
├── configs/       paths, datasets, preprocessing and baseline hyperparameters (YAML)
├── data/          gitignored - see data/README.md
├── outputs/       gitignored - checkpoints, heatmaps, results
├── docs/          proposal, calendar, workplan, bootstrap cell
├── dashboard/     FYP-II placeholder
├── README.md  requirements.txt  .gitignore
```

## Team workflow

Code travels by GitHub; data and checkpoints travel by the shared Drive folder `XRetina_Shared/`.

1. **Branches:** work on your own branch (`aiman/data`, `maheen/trainer`, `amna/baseline`). Do not commit straight to `main`.
2. **Pull requests:** finish a step with a PR into `main` using the template. The next person starts after it is merged.
3. **Start of every session:** run the bootstrap cell (it clones or pulls the repo and installs requirements).
4. **Push from Colab:** use `git_push("message")` (defined by the bootstrap cell). It refuses to push to `main`.
5. **Notebook hygiene:** clear large outputs before committing. Optional: `pip install nbstripout && nbstripout --install` to strip outputs automatically and avoid merge conflicts.

## One-time setup per member

1. Accept the GitHub invite.
2. Make the shared Drive folder visible: add a shortcut named `XRetina_Shared` to your My Drive.
3. Colab > left sidebar > Secrets (key icon), add and enable notebook access for: `GITHUB_TOKEN` (personal access token with repo access), `KAGGLE_USERNAME`, `KAGGLE_KEY`.
4. Open `notebooks/00_colab_setup.ipynb` in Colab and run it once to confirm everything works.

## FYP-I calendar pace

| Week | Dates | Milestone |
|---|---|---|
| 8 | 5-11 Oct | Data download + preprocessing |
| 9 | 12-18 Oct | Dataloaders + trainer |
| 10 | 19-25 Oct | Working prototype / intermediate results (graded) |
| 11-12 | 26 Oct-8 Nov | Mid-term evaluation 6-8 Nov (20 marks) |
| 16 | 30 Nov-4 Dec | Complete draft |
| Finals | 15-26 Dec | Final report |

Source: `docs/calendar/`. Tracker: `docs/planning/XRetina_FYP-I_Team_Workplan.xlsx`.

## Note on docs

`docs/proposal/` contains the proposal with personal contact details. Keep this repo **private**, or remove that file before ever making it public.
