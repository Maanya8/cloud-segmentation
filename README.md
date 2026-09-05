# Cloud Segmentation on Landsat-8 Imagery

This project trains a binary cloud-segmentation model on the 38-Cloud Landsat-8 dataset. It uses four spectral input channels (red, green, blue, and near-infrared), a pretrained MobileNetV2 encoder, and an FPN-style decoder.

## Requirements

- Windows with Python 3.10 or newer
- The 38-Cloud training data in the repository's `Cloud38` directory
- A CUDA-capable GPU is recommended, but the script automatically falls back to CPU

## Setup

From the project directory, run:

```bat
setup.bat
```

This creates `.venv`, upgrades `pip`, and installs the packages in `requirements.txt`.

## Dataset layout

The default configuration expects these paths:

```text
Cloud38/
  training_patches_38-cloud_nonempty.csv
  38-Cloud_training/
    train_red/
    train_green/
    train_blue/
    train_nir/
    train_gt/
  38-Cloud_Training_Metadata_Files/
    38-Cloud_Training_Metadata_Files/
```

The metadata directory must contain the Landsat scene files named like `<scene_id>_MTL.txt`. The non-empty patch CSV prevents training on patches dominated by no-data margins.

## Train

Activate the environment and start training:

```bat
.venv\Scripts\activate
python training.py
```

The default run trains for 61 epochs, uses a 15% validation split, and writes outputs to:

- `checkpoints_single/best_model.pt`
- `checkpoints_single/last_checkpoint.pt`
- `epoch_visualizations/epoch_XXX/`

To continue an interrupted run from the last checkpoint:

```bat
python training.py --resume true
```

Any configuration field in `training.py` can be overridden from the command line. For example:

```bat
python training.py --batch_size 8 --num_epochs 10 --downsample_factor 2
```

Use `--device cpu` to force CPU training.

## Model and evaluation

The model predicts one cloud logit per pixel. Training combines binary cross-entropy and Dice loss. Validation reports loss, accuracy, precision, recall, F1, IoU, Dice, and cloud-pixel fraction. The script also saves a fixed validation example after each epoch for visual inspection.

## Project notes

Additional project rationale, dataset details, and the original experiment plan are documented in [steps.md](steps.md).