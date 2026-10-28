# ICBHI 2017 Binary Classification with PretrainedSED

This project adapts the PretrainedSED audio-transformer codebase for respiratory-sound classification on the ICBHI 2017 Respiratory Sound Database. The current two-class training script uses pretrained audio encoders and predicts **normal** versus **abnormal** respiratory cycles. It trains on the training portion, makes a patient-disjoint training/validation split, selects the checkpoint with the best validation ICBHI HS score, and evaluates that checkpoint on the held-out test portion.

The upstream project provides pretrained sound-event-detection architectures and weights. This thesis adaptation changes the prediction head, dataset loader, training loss, augmentations, and evaluation metrics for clip-level binary classification.

## 1. Requirements

- Windows, Linux, or macOS
- Python **3.11.9** (the version declared in `pyproject.toml`)
- Poetry
- A CUDA-capable GPU is recommended for practical training; CPU execution may be slow
- A Weights & Biases account and login, because the training script creates a W&B logger

Install Poetry using the [official Poetry instructions](https://python-poetry.org/docs/#installation). On Windows PowerShell, the official installer command is:

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

Open a new terminal after installation, change to the project directory, and install the locked project dependencies:

```powershell
poetry env use 3.11.9
poetry install
```

The project declares PyTorch audio and vision packages through `pyproject.toml`. For GPU training, ensure that the installed PyTorch build matches the CUDA version supported by your system; use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) if you need a specific CUDA build. You can check whether CUDA is available in the Poetry environment with:

```powershell
poetry run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 2. Get and prepare the ICBHI dataset

Download the Respiratory Sound Database from the [official ICBHI Challenge website](https://bhichallenge.med.auth.gr/). The dataset is provided for research use; follow the website's terms and cite the dataset in research outputs. The website may require registration for access to some challenge materials.

The training script does not consume the raw recording/annotation archive directly. It expects respiratory-cycle audio files and a prepared metadata CSV in this layout:

```text
ICBHI_dataset/
├── meta.csv
└── audios_cycles/
    ├── <cycle-audio-file>.wav
    └── ...
```

`meta.csv` must contain at least these columns:

| Column | Meaning |
| --- | --- |
| `fold` | `train` or `test`; the official test rows stay held out |
| `filename` | Audio filename located under `ICBHI_dataset/audios_cycles/` |
| `target` | Binary class ID: `0` for normal and `1` for abnormal |
| `label` | Optional four-class label; the current binary training command reads `target` |

Use one row per respiratory cycle, and make sure each filename is unique and matches an audio file. Patient IDs are inferred from the filename prefix before the first underscore; cycles from a patient are therefore grouped together when the script creates its validation split. The dataset loader resamples audio to 16 kHz and pads or truncates each cycle to 10 seconds.

The repository's preprocessing code or notebook should be used to create this cycle-level layout and CSV from the downloaded recordings and annotations. Keep the official test partition marked `test`; the script divides only rows marked `train` into train and validation groups.

## 3. Configure Weights & Biases

The training script logs metrics to the W&B project `PTSED`. Authenticate once in the terminal:

```powershell
poetry run wandb login
```

Follow the prompt to enter your W&B API key. Training logs include loss and validation metrics; the best validation HS-score is used to select a checkpoint.

## 4. Train a model

Run commands from the repository root so the relative dataset paths resolve:

```powershell
poetry run python ex_audioset_strong_2cls_May1-Aug24.py `
  --model_name ATST-F `
  --pretrained weak `
  --n_epochs 30 `
  --batch_size 32 `
  --num_workers 4
```

The script supports these `--model_name` values: `ATST-F`, `BEATs`, `fpasst`, `M2D`, `ASIT`, `frame_mn06`, and `frame_mn10`. The `--pretrained` choices are `scratch`, `ssl`, `weak`, and `strong`. Adjust batch size and worker count to suit your hardware. The default script uses automatic accelerator selection and validates each epoch.

After training, Lightning saves the checkpoint with the highest `val/HS_score` and evaluates that best checkpoint on the official test split. Checkpoints and logs are written by Lightning/W&B; retain the run information if you need to reproduce a result.

For a shorter run, reduce `--n_epochs`. To resume or compare experiments, provide a distinct run name with `--experiment_name`.

## 5. Important files

- `ex_audioset_strong.py` — upstream PretrainedSED AudioSet-Strong training script
- `ex_audioset_weak_2cls_Ver01_commented.py` — earlier two-class adaptation
- `ex_audioset_weak_2cls_Ver02_commented.py` — Version 02 adaptation with best-checkpoint reporting
- `ex_audioset_weak_2cls_Ver03_commented.py` — Version 03 patient-level train/validation split and held-out test evaluation
- `ICBHI_dataset/audios` - this directory must be created to hold only the all 920 original audios (*.wav)
- `ICBHI_dataset/metadata_df.csv` - metadata consumed by the preprocessing step to extract audios at the cycle level and save them into `ICBHI_dataset/audios_cycles/`
- `ICBHI_dataset/splitting to audio cycles.ipynb` - the Notebook is used to extract and save audio cycles from the 920 original audios
- `ICBHI_dataset/meta.csv` — cycle-level metadata consumed by the August 24 script
- `ICBHI_dataset/audios_cycles/` — cycle-level audio referenced by `meta.csv`
- `pyproject.toml` and `poetry.lock` — Poetry dependencies and locked versions

## 6. Reproducibility notes

- Keep a copy of the exact metadata CSV and preprocessing settings used for each experiment.
- The validation split is grouped by patient with a fixed random seed (`42`); changing metadata or code can change the split.
- Keep test data out of training and model selection. The script uses the `test` rows only after selecting the best validation checkpoint.
- Record the encoder, pretrained checkpoint type, batch size, number of epochs, and W&B run when reporting results.
- The script's defaults and checkpoint availability may depend on network access and the pretrained model resources configured by the upstream project.

## References

- [ICBHI Challenge and Respiratory Sound Database](https://bhichallenge.med.auth.gr/)
- [PretrainedSED: Effective Pre-Training of Audio Transformers for Sound Event Detection](https://github.com/fschmid56/PretrainedSED)
- [Poetry installation and usage documentation](https://python-poetry.org/docs/)
