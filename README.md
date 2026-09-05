# Cloud Segmentation on Landsat-8 Imagery

This project trains a binary cloud-segmentation model on the 38-Cloud Landsat-8 dataset. It uses four spectral input channels (red, green, blue, and near-infrared), a pretrained MobileNetV2 encoder, and an FPN-style decoder. The project emphasizes reliable validation and confidence-aware inspection rather than accuracy alone.

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

## Dataset

The 38-Cloud dataset contains 38 Landsat-8 scenes, approximately 8,400 training patches, and 9,201 test patches. Each patch is 384 x 384 pixels and stores the spectral bands as separate single-band TIFF files:

| Input channel | Landsat band |
| --- | ---: |
| Red | 4 |
| Green | 3 |
| Blue | 2 |
| Near-infrared | 5 |

The ground truth is a binary per-pixel cloud mask. Many source patches contain black no-data margins because Landsat scenes are diagonal swaths in rectangular products, so training uses the supplied non-empty patch list.

For each patch, raw digital numbers are converted to top-of-atmosphere reflectance using the source scene's MTL coefficients and sun elevation, then clipped to `[0, 1]`. This keeps inputs comparable across scenes.

To replicate, download the Cloud38 dataset, as on https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images, and put it in root directory of the project in a folder named Cloud38

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

## Model and training

The model predicts one cloud logit per pixel. Its pretrained MobileNetV2 encoder is adapted from 3 input channels to 4, and an FPN/U-Net-style multi-resolution decoder produces the segmentation map. The output is converted with a sigmoid into per-pixel cloud probabilities.

Training combines binary cross-entropy with Dice loss. The backbone is initially frozen and is fine-tuned after the configured warm-up period. Training patches use random horizontal and vertical flips; validation patches are left unchanged.

## Evaluation and confidence

Validation reports loss, accuracy, precision, recall, F1, IoU, Dice, and cloud-pixel fraction. IoU and Dice are the primary overlap metrics for cloud masks. A fixed validation example is saved after each epoch with the input, ground truth, and prediction for visual inspection.

The sigmoid output can also be used to create confidence maps. Confidence may be represented as distance from the 0.5 decision threshold or with predictive entropy. A stronger final evaluation should stitch patch predictions back into complete scenes before comparing them with the full-scene test ground truth.

## Scope and limitations

This project demonstrates cloud-mask generation on real Level-1 Landsat-8 imagery, PyTorch segmentation, multi-channel remote-sensing inputs, and validation-oriented reporting. It does not by itself provide separate shadow, snow, water, or haze classes, hyperspectral processing, denoising, destriping, super-resolution, or calibrated confidence curves. Those are possible extensions rather than outputs of the current training script.

For a portfolio description, this can be framed as adapting an encoder-decoder architecture originally intended for keypoint heatmap regression to pixel-level cloud segmentation, with emphasis on confidence estimation and scene-level QA/QC workflows.