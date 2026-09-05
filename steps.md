Project Title

Cloud Segmentation on Landsat-8 Imagery (38-Cloud Dataset)

Objectives

Train a deep learning model to perform pixel-level binary segmentation (cloud vs. not-cloud) on Landsat-8 satellite imagery, with an added focus on validation rigor — confidence estimation and proper evaluation methodology — rather than just raw accuracy.

Dataset
38-Cloud: 38 Landsat-8 scenes, pre-cut into 384×384 patches — 8400 training patches, 9201 test patches.
4 spectral channels per patch: Red (band 4), Green (band 3), Blue (band 2), Near-Infrared (band 5) — stored as separate single-band TIFFs per patch, so your data loader needs to stack them into one 4-channel tensor.
Ground truth: pixel-level binary cloud masks. Test-set ground truth is provided at full-scene level (not per-patch), requiring a stitching step for final evaluation.
Known data quirk: many patches are partially/fully black due to no-data margins from Landsat's diagonal swath vs. rectangular product format. Filter using the provided training_patches_38-cloud_nonempty.csv (patches with >80% informative pixels) to avoid training on empty regions.
Architecture

Reuse the encoder-decoder backbone from your keypoint pose estimation project, repurposed for segmentation:

Encoder: MobileNetV2 (pretrained), input channels changed from 3 → 4 to accommodate R/G/B/NIR.
Decoder: your existing FPN-style multi-resolution fusion decoder — same pattern as before, just outputting a segmentation map instead of keypoint heatmaps.
Output head: single-channel per-pixel logit → sigmoid → binary cloud probability map (since this is binary segmentation, not multi-class).
(Optional, if you want to keep the multi-task pattern from your keypoint project alive): a secondary head predicting per-patch "informative vs. margin/no-data" — mirrors your visibility-logit branch conceptually, though it's not essential.
Loss Function
BCE + Dice loss combination — standard for binary segmentation; Dice helps with thin/small cloud regions that pure BCE tends to under-weight.
Optionally add focal loss if you observe heavy imbalance between cloud/no-cloud pixels in your training split.
Training Setup (laptop-friendly)
Patch size 384×384 (or downsize to 192×192 to speed up training, as the original Cloud-Net paper did).
Small batch size (8–16) suitable for CPU/single-GPU laptop.
Standard augmentations: flips, rotations, brightness/contrast jitter (simulates illumination variation across scenes).
Evaluation
Primary metric: IoU / Dice coefficient on the test set.
Scene-level stitching: reassemble patch predictions into full scenes before comparing to ground truth, following the dataset's documented evaluation procedure — this is a nice production-realistic touch most portfolio projects skip.
Confidence maps: derive per-pixel confidence from the sigmoid output (distance from 0.5, or entropy) and visualize alongside predictions.
Calibration check (stretch goal): plot predicted confidence vs. actual accuracy to show whether the model's confidence is trustworthy, not just its raw accuracy.
What This Project Demonstrates (mapped to the JD)

Solidly covered:

Cloud mask generation on real L1-level Landsat-8 imagery — directly matches part of the "cloud, water, shadow, snow, haze, unusable-pixel masks" responsibility.
Validation framework + confidence scoring — explicitly named in "success looks like."
Core required skills: PyTorch, OpenCV, NumPy, segmentation, working with large real-world satellite datasets.

Not covered by this project alone (per our earlier breakdown): shadow/snow/water/haze-specific classes, hyperspectral data, denoising/destriping, super-resolution, biome/condition-pattern analysis, LLM-based operational tooling.

Suggested Write-up Framing

Position it as: "Adapted a shared encoder-decoder architecture (originally built for multi-task keypoint heatmap regression) to pixel-level cloud segmentation on real Landsat-8 imagery, with emphasis on confidence calibration and scene-level evaluation matching production QA/QC workflows." This ties it explicitly back to your existing project and to the JD's own language around validation rigor.