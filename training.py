import argparse
import csv
import math
import random
import re
from dataclasses import dataclass, fields, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Config
# ============================================================
@dataclass
class Config:

    # --- Cloud38 data paths ---
    # CSV listing the non-empty (informative) training patch names, e.g.
    # "patch_1_1_by_10_LC08_L1TP_002053_20160520_20170324_01_T1" -- no channel
    # prefix (red_/green_/blue_/nir_/gt_) and no file extension.
    cloud38_csv: str = "./Cloud38/training_patches_38-cloud_nonempty.csv"
    # Root folder containing train_red/, train_green/, train_blue/, train_nir/, train_gt/
    cloud38_training_dir: str = "./Cloud38/38-Cloud_training"
    # Folder of per-scene Landsat MTL.txt files (REFLECTANCE_MULT/ADD_BAND_X,
    # SUN_ELEVATION), used to convert raw pixel DNs to TOA reflectance.
    cloud38_metadata_dir: str = "./Cloud38/38-Cloud_Training_Metadata_Files/38-Cloud_Training_Metadata_Files"
    checkpoint_dir: str = "./checkpoints_single"
    # Per-epoch visualization: one fixed validation patch's input/ground-truth/
    # prediction is saved here as PNGs, in a subfolder per epoch.
    visualization_dir: str = "./epoch_visualizations"
    visualization_min_cloud_fraction: float = 0.30
    visualization_max_cloud_fraction: float = 0.80

    # --- data shape ---
    # 38-Cloud patches are all 384x384. downsample_factor divides both
    # dimensions (2 -> 192x192). See the accompanying explanation for why 2
    # is a reasonable default.
    patch_size: int = 384
    downsample_factor: int = 2
    num_input_channels: int = 4    # red, green, blue, nir
    num_output_classes: int = 1    # single binary cloud/no-cloud channel

    # --- split ---
    val_split: float = 0.15
    split_seed: int = 42

    # --- model ---
    pretrained_backbone: bool = True
    fpn_channels: int = 128        # channel width used throughout the FPN decoder's lateral/smoothing convs

    # --- training schedule ---
    batch_size: int = 32
    num_epochs: int = 61
    freeze_epochs: int = 20        # epochs to keep MobileNetV2 backbone frozen before fine-tuning
    lr_frozen: float = 1e-3        # LR while backbone is frozen (head-only training)
    lr_finetune: float = 1e-4      # LR after unfreezing (whole network)
    weight_decay: float = 1e-4

    # --- early stopping ---
    early_stopping_patience: int = 8

    # --- resuming ---
    resume: bool = False           # if True, resume from checkpoint_dir/last_checkpoint.pt if it exists

    # --- misc ---
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Train Cloud38 Segmentation Model")
    for f in fields(cfg):
        default = getattr(cfg, f.name)
        arg_type = type(default) if default is not None else str
        if f.type == "bool" or isinstance(default, bool):
            parser.add_argument(f"--{f.name}", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=default)
        else:
            parser.add_argument(f"--{f.name}", type=arg_type if default is not None else str, default=default)
    args = parser.parse_args()
    for f in fields(cfg):
        setattr(cfg, f.name, getattr(args, f.name))
    return cfg


# ============================================================
# 2. Cloud38 patch discovery + input/mask loading
#
#     38-Cloud on-disk layout:
#       38-Cloud_training/train_<channel>/<channel>_<patch_name>.TIF
#     with channel in {red, green, blue, nir, gt}.
# ============================================================
CLOUD38_DIRS = {
    "red": "train_red",
    "green": "train_green",
    "blue": "train_blue",
    "nir": "train_nir",
    "gt": "train_gt",
}
# Stack order used when building the 4-channel input tensor.
CLOUD38_INPUT_CHANNELS = ("red", "green", "blue", "nir")


def load_nonempty_patch_names(csv_path):
    """
    Reads training_patches_38-cloud_nonempty.csv and returns the list of
    patch name stems it contains, e.g.
    "patch_1_1_by_10_LC08_L1TP_002053_20160520_20170324_01_T1".

    The CSV is assumed to have a header row followed by one patch name per
    row in the first column; the header's exact wording isn't relied on.
    """
    patch_names = []
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.reader(f))

    for row in rows[1:]:  # skip header
        if not row:
            continue
        name = row[0].strip()
        if name:
            patch_names.append(name)
    return patch_names


def build_cloud38_train_val_ids(csv_path, val_split, seed):
    all_ids = load_nonempty_patch_names(csv_path)
    rng = random.Random(seed)
    rng.shuffle(all_ids)

    n_val = max(1, int(len(all_ids) * val_split))
    val_ids = all_ids[:n_val]
    train_ids = all_ids[n_val:]
    return train_ids, val_ids


def _find_patch_file(training_dir, prefix, patch_name):
    """Locates the on-disk .TIF file for one channel/mask of one patch."""
    channel_dir = Path(training_dir) / CLOUD38_DIRS[prefix]
    for ext in (".TIF", ".tif"):
        candidate = channel_dir / f"{prefix}_{patch_name}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a '{prefix}' patch file for '{patch_name}' in {channel_dir} "
        f"(looked for {prefix}_{patch_name}.TIF / .tif)"
    )


# --- reflectance normalization, driven by each scene's MTL.txt ---
# 38-Cloud patch pixels are the raw Landsat 8 digital numbers (DN), not
# reflectance. Converting DN -> top-of-atmosphere (TOA) reflectance using the
# scene's own MTL coefficients (rather than a flat /65535) gives values on a
# physically meaningful, roughly-comparable-across-scenes [0, 1] scale.
#
# Formula (per the USGS Landsat 8 handbook), sun-angle corrected:
#   reflectance = (REFLECTANCE_MULT_BAND_X * DN + REFLECTANCE_ADD_BAND_X) / sin(SUN_ELEVATION)
#
# Band numbers for our 4 channels (Landsat 8 OLI band numbering):
CLOUD38_CHANNEL_TO_LANDSAT_BAND = {"red": 4, "green": 3, "blue": 2, "nir": 5}

_PATCH_NAME_RE = re.compile(r"^patch_\d+_\d+_by_\d+_(.+)$")

# Cache of parsed MTL files, keyed by (metadata_dir, scene_id) -- many patches
# share the same source scene, so this avoids re-reading/re-parsing the same
# .txt file for every single patch.
_MTL_CACHE = {}


def _scene_id_from_patch_name(patch_name):
    """
    38-Cloud patch names look like "patch_<row>_<col>_by_<n>_<scene_id>",
    e.g. "patch_1_1_by_10_LC08_L1TP_002053_20160520_20170324_01_T1". This
    strips the "patch_<row>_<col>_by_<n>_" prefix to recover the scene id,
    which is also the MTL file's basename (<scene_id>_MTL.txt).
    """
    match = _PATCH_NAME_RE.match(patch_name)
    if not match:
        raise ValueError(
            f"Could not parse a Landsat scene id out of patch name '{patch_name}' "
            f"(expected 'patch_<row>_<col>_by_<n>_<scene_id>')"
        )
    return match.group(1)


def _parse_mtl_file(path):
    """Parses a Landsat MTL.txt into a flat {KEY: value_string} dict."""
    values = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip().strip('"')
    return values


def _get_mtl_values(metadata_dir, scene_id):
    cache_key = (str(metadata_dir), scene_id)
    if cache_key not in _MTL_CACHE:
        path = Path(metadata_dir) / f"{scene_id}_MTL.txt"
        _MTL_CACHE[cache_key] = _parse_mtl_file(path)
    return _MTL_CACHE[cache_key]


def _dn_to_toa_reflectance(dn, mtl_values, landsat_band):
    mult = float(mtl_values[f"REFLECTANCE_MULT_BAND_{landsat_band}"])
    add = float(mtl_values[f"REFLECTANCE_ADD_BAND_{landsat_band}"])
    sun_elevation_deg = float(mtl_values["SUN_ELEVATION"])
    sun_elevation_rad = math.radians(sun_elevation_deg)
    reflectance = (mult * dn + add) / math.sin(sun_elevation_rad)
    # TOA reflectance can slightly exceed [0, 1] (per this scene's own
    # MIN_MAX_REFLECTANCE values) or dip slightly negative in dark/low-signal
    # pixels; clip to [0, 1] so the model always sees a bounded input.
    return np.clip(reflectance, 0.0, 1.0)


def load_cloud38_patch_image(patch_name, training_dir, metadata_dir):
    """
    Loads the four spectral bands (red, green, blue, nir) for one 38-Cloud
    patch, converts each from raw DN to TOA reflectance using that patch's
    source scene MTL file, and stacks them into a single (4, H, W) float32
    array in CLOUD38_INPUT_CHANNELS order.
    """
    scene_id = _scene_id_from_patch_name(patch_name)
    mtl_values = _get_mtl_values(metadata_dir, scene_id)

    band_arrays = []
    for channel in CLOUD38_INPUT_CHANNELS:
        path = _find_patch_file(training_dir, channel, patch_name)
        dn = np.array(Image.open(path), dtype=np.float32)
        landsat_band = CLOUD38_CHANNEL_TO_LANDSAT_BAND[channel]
        reflectance = _dn_to_toa_reflectance(dn, mtl_values, landsat_band)
        band_arrays.append(reflectance)
    return np.stack(band_arrays, axis=0)  # (4, H, W)


def load_cloud38_patch_mask(patch_name, training_dir):
    """
    Loads the ground-truth mask for one patch and binarizes it: any nonzero
    pixel (typically stored as 255) is treated as "cloud" (1.0), all others
    as "no cloud" (0.0). Returns a (H, W) float32 array.
    """
    path = _find_patch_file(training_dir, "gt", patch_name)
    raw = np.array(Image.open(path))
    mask = (raw > 0).astype(np.float32)
    return mask


class Cloud38SegmentationDataset(Dataset):
    """
    For each non-empty patch name (from training_patches_38-cloud_nonempty.csv),
    loads the matching red/green/blue/nir .TIF files as the input image
    (converted DN -> TOA reflectance via the patch's scene MTL file) and the
    matching train_gt .TIF file as the binary target mask.

    downsample_factor: if > 1, both image and mask are downsampled by that
    factor (image via bilinear interpolation, mask via nearest-neighbor so it
    stays exactly binary).

    augment: if True, applies random horizontal/vertical flips (identically
    to image and mask). Satellite patches have no canonical "up", so flips
    are a safe, label-preserving augmentation; brightness/contrast jitter
    (used for the old RGB task) doesn't make sense for raw reflectance bands
    and has been dropped.
    """

    def __init__(self, patch_names, training_dir, metadata_dir, downsample_factor=1, augment=False):
        self.patch_names = patch_names
        self.training_dir = Path(training_dir)
        self.metadata_dir = Path(metadata_dir)
        self.downsample_factor = downsample_factor
        self.augment = augment

    def __len__(self):
        return len(self.patch_names)

    def __getitem__(self, idx):
        patch_name = self.patch_names[idx]

        image = torch.from_numpy(
            load_cloud38_patch_image(patch_name, self.training_dir, self.metadata_dir)
        )  # (4, H, W)
        mask = torch.from_numpy(load_cloud38_patch_mask(patch_name, self.training_dir)).unsqueeze(0)  # (1, H, W)

        if self.downsample_factor > 1:
            scale = 1.0 / self.downsample_factor
            image = nn.functional.interpolate(
                image.unsqueeze(0), scale_factor=scale, mode="bilinear",
                align_corners=False, recompute_scale_factor=False,
            ).squeeze(0)
            mask = nn.functional.interpolate(
                mask.unsqueeze(0), scale_factor=scale, mode="nearest",
                recompute_scale_factor=False,
            ).squeeze(0)

        if self.augment:
            if random.random() < 0.5:
                image = torch.flip(image, dims=[-1])
                mask = torch.flip(mask, dims=[-1])
            if random.random() < 0.5:
                image = torch.flip(image, dims=[-2])
                mask = torch.flip(mask, dims=[-2])

        return image, mask


# ============================================================
# 3. Model
# ============================================================
class MobileNetSegmentationNet(nn.Module):
    """
    MobileNetV2 backbone with an FPN/U-Net-style decoder: instead of decoding
    purely from the deepest (stride-32) feature map, this fuses in
    higher-resolution intermediate features via lateral skip connections
    (stride 4, 8, 16, 32), so fine spatial detail is preserved. Outputs a
    single-channel logit map upsampled to `target_size`, i.e. one cloud/
    no-cloud score per input pixel (after `downsample_factor`).
    """

    SKIP_LAYERS = {
        3: (24, 4),     # stride 4  -- highest resolution we use
        6: (32, 8),     # stride 8
        13: (96, 16),   # stride 16
        18: (1280, 32), # stride 32 -- deepest features
    }

    def __init__(self, in_channels=4, num_output_classes=1, pretrained=True,
                 fpn_channels=128, target_size=(192, 192)):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V2" if pretrained else None)
        self.backbone_layers = nn.ModuleList(list(mobilenet.features.children()))

        # --- adapt the first conv to accept `in_channels` inputs instead of
        # the ImageNet-pretrained 3 (RGB). We keep the pretrained RGB weights
        # for the first 3 input channels, and initialize any extra channels
        # (e.g. NIR) as the average of the pretrained RGB kernels -- this
        # keeps the layer's output statistics close to what it was trained
        # for, so the pretrained low-level filters (edges/textures) are
        # still useful instead of that layer starting from scratch. ---
        if in_channels != 3:
            old_conv = self.backbone_layers[0][0]  # nn.Conv2d(3, 32, k=3, s=2, p=1, bias=False)
            new_conv = nn.Conv2d(
                in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None,
            )
            if pretrained:
                with torch.no_grad():
                    n_copy = min(3, in_channels)
                    new_conv.weight[:, :n_copy] = old_conv.weight[:, :n_copy]
                    if in_channels > 3:
                        mean_kernel = old_conv.weight.mean(dim=1, keepdim=True)  # (32, 1, kh, kw)
                        new_conv.weight[:, 3:] = mean_kernel.repeat(1, in_channels - 3, 1, 1)
            self.backbone_layers[0][0] = new_conv

        self.target_size = target_size
        self.deepest_idx = max(self.SKIP_LAYERS.keys())

        self.lateral_convs = nn.ModuleDict({
            str(idx): nn.Conv2d(ch, fpn_channels, kernel_size=1)
            for idx, (ch, _stride) in self.SKIP_LAYERS.items()
        })
        self.smooth_convs = nn.ModuleDict({
            str(idx): nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
            for idx in self.SKIP_LAYERS if idx != self.deepest_idx
        })

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, num_output_classes, kernel_size=1),
        )

    def forward(self, x):
        skip_feats = {}
        h = x
        for i, layer in enumerate(self.backbone_layers):
            h = layer(h)
            if i in self.SKIP_LAYERS:
                skip_feats[i] = h
            if i == self.deepest_idx:
                break

        sorted_idx = sorted(self.SKIP_LAYERS.keys(), reverse=True)  # [18, 13, 6, 3]
        fused = self.lateral_convs[str(sorted_idx[0])](skip_feats[sorted_idx[0]])
        for idx in sorted_idx[1:]:
            lateral = self.lateral_convs[str(idx)](skip_feats[idx])
            fused_upsampled = nn.functional.interpolate(
                fused, size=lateral.shape[-2:], mode="bilinear", align_corners=False
            )
            fused = self.smooth_convs[str(idx)](fused_upsampled + lateral)
        # `fused` is at stride 4

        logits = self.segmentation_head(fused)  # (B, num_output_classes, stride4_H, stride4_W)
        logits = nn.functional.interpolate(
            logits, size=self.target_size, mode="bilinear", align_corners=False
        )
        return logits  # raw logits -- apply sigmoid outside for probabilities

    # --- freeze / unfreeze helpers (BatchNorm-aware) ---
    def freeze_backbone(self):
        for layer in self.backbone_layers:
            for param in layer.parameters():
                param.requires_grad = False
            layer.eval()  # freezes BatchNorm running stats too

    def unfreeze_backbone(self):
        for layer in self.backbone_layers:
            for param in layer.parameters():
                param.requires_grad = True
            layer.train()


# ============================================================
# 4. Loss
# ============================================================
def dice_loss(probs, target, eps=1e-6):
    probs = probs.reshape(probs.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def segmentation_loss(logits, target_mask):
    """
    BCE (on raw logits, for numerical stability) + Dice. BCE gives a stable
    per-pixel gradient; Dice optimizes directly for mask overlap, which
    helps when cloud/no-cloud pixels are imbalanced and sharpens edges.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target_mask)
    dice = dice_loss(torch.sigmoid(logits), target_mask)
    return bce + dice


# ============================================================
# 5. Validation metrics
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    """
    Runs the full validation set and returns loss plus standard binary
    segmentation metrics, computed from pixel counts accumulated over the
    *entire* validation set (not averaged per-batch), so the ratio metrics
    aren't biased by variable-sized or unevenly-imbalanced batches.
    """
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0.0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = segmentation_loss(logits, masks)
        total_loss += loss.item()

        preds = (torch.sigmoid(logits) > threshold).float()
        tp += (preds * masks).sum().item()
        fp += (preds * (1 - masks)).sum().item()
        fn += ((1 - preds) * masks).sum().item()
        tn += ((1 - preds) * (1 - masks)).sum().item()

    eps = 1e-7
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    # Note: for binary masks, Dice and F1 are mathematically the same
    # quantity -- both reported since each name is the conventional one in
    # different communities (segmentation vs. classification).
    dice = 2 * tp / (2 * tp + fp + fn + eps)

    return {
        "val_loss": total_loss / len(loader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice,
        "cloud_pixel_fraction": (tp + fn) / (tp + tn + fp + fn + eps),  # how much of val set is actually cloud
    }


# ============================================================
# 6. Per-epoch visualization
# ============================================================
def _reflectance_to_uint8_rgb(image_chw):
    """
    image_chw: (4, H, W) tensor of TOA reflectance in [0, 1], channel order
    red/green/blue/nir (see CLOUD38_INPUT_CHANNELS). Takes the first 3
    channels as a natural-color preview and converts to uint8 (H, W, 3).
    """
    rgb = image_chw[:3].detach().cpu().clamp(0, 1).numpy()
    rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)
    return (rgb * 255).astype(np.uint8)


def _prob_to_uint8_mask(mask_hw):
    """mask_hw: (H, W) or (1, H, W) tensor in [0, 1] -> uint8 (H, W) grayscale."""
    arr = mask_hw.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)


def save_epoch_visualization(epoch, image, true_mask, pred_probs, output_dir):
    """
    Saves one fixed validation patch's input image, ground-truth mask, and
    the model's current predicted probability map as PNGs in
    output_dir/epoch_<NNN>/, plus a single side-by-side "combined.png" for
    quick visual tracking of how predictions change epoch to epoch.
    """
    epoch_dir = Path(output_dir) / f"epoch_{epoch + 1:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    rgb_img = Image.fromarray(_reflectance_to_uint8_rgb(image))
    true_img = Image.fromarray(_prob_to_uint8_mask(true_mask)).convert("RGB")
    pred_img = Image.fromarray(_prob_to_uint8_mask(pred_probs)).convert("RGB")

    rgb_img.save(epoch_dir / "input_rgb.png")
    true_img.save(epoch_dir / "ground_truth_mask.png")
    pred_img.save(epoch_dir / "predicted_mask.png")

    w, h = rgb_img.size
    combined = Image.new("RGB", (w * 3, h))
    combined.paste(rgb_img, (0, 0))
    combined.paste(true_img, (w, 0))
    combined.paste(pred_img, (w * 2, 0))
    combined.save(epoch_dir / "combined.png")  # left-to-right: input | ground truth | prediction


def select_visualization_patch(val_ids, training_dir, min_cloud_fraction, max_cloud_fraction):
    """Selects the first validation patch within the requested cloud range."""
    for patch_name in val_ids:
        mask = load_cloud38_patch_mask(patch_name, training_dir)
        cloud_fraction = float(mask.mean())
        if min_cloud_fraction <= cloud_fraction <= max_cloud_fraction:
            return patch_name, cloud_fraction

    raise ValueError(
        f"No validation patch has cloud coverage between "
        f"{min_cloud_fraction:.0%} and {max_cloud_fraction:.0%}."
    )


# ============================================================
# 7. Resumable-training checkpoint helpers
# ============================================================
def save_resume_checkpoint(path, epoch, model, optimizer, best_val_loss, epochs_without_improvement, cfg):
    """Saves everything needed to resume training exactly where it left off."""
    torch.save({
        "epoch": epoch,  # last COMPLETED epoch index (0-based)
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "config": asdict(cfg),
    }, path)


def load_resume_checkpoint(path, device):
    # weights_only=False: this checkpoint is self-generated (not third-party) and
    # includes optimizer/RNG state alongside tensors, which PyTorch's default
    # weights_only=True loader (as of PyTorch 2.6+) does not support deserializing.
    return torch.load(path, map_location=device, weights_only=False)


def restore_rng_state(ckpt):
    torch.set_rng_state(ckpt["torch_rng_state"])
    if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    np.random.set_state(ckpt["numpy_rng_state"])
    random.setstate(ckpt["python_rng_state"])


# ============================================================
# 8. Training loop
# ============================================================
def train(cfg: Config):
    set_seed(cfg.seed)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    resume_path = Path(cfg.checkpoint_dir) / "last_checkpoint.pt"

    train_ids, val_ids = build_cloud38_train_val_ids(cfg.cloud38_csv, cfg.val_split, cfg.split_seed)
    print(f"Train patches: {len(train_ids)} | Val patches: {len(val_ids)}")

    target_size = (cfg.patch_size // cfg.downsample_factor, cfg.patch_size // cfg.downsample_factor)
    print(f"Input/output resolution: {target_size[0]}x{target_size[1]} "
          f"(native {cfg.patch_size}x{cfg.patch_size}, downsample_factor={cfg.downsample_factor})")

    train_ds = Cloud38SegmentationDataset(
        train_ids, cfg.cloud38_training_dir, cfg.cloud38_metadata_dir,
        downsample_factor=cfg.downsample_factor, augment=True,
    )
    val_ds = Cloud38SegmentationDataset(
        val_ids, cfg.cloud38_training_dir, cfg.cloud38_metadata_dir,
        downsample_factor=cfg.downsample_factor, augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    # Fixed sample used for the per-epoch visualization snapshot. Select it by
    # ground-truth cloud coverage once so every epoch remains comparable.
    viz_patch_name, viz_cloud_fraction = select_visualization_patch(
        val_ids, cfg.cloud38_training_dir,
        cfg.visualization_min_cloud_fraction, cfg.visualization_max_cloud_fraction,
    )
    viz_index = val_ids.index(viz_patch_name)
    viz_image, viz_true_mask = val_ds[viz_index]
    print(f"Visualization patch: {viz_patch_name} "
            f"(cloud coverage={viz_cloud_fraction:.2%}, "
            f"required={cfg.visualization_min_cloud_fraction:.0%}-"
            f"{cfg.visualization_max_cloud_fraction:.0%})")
    Path(cfg.visualization_dir).mkdir(parents=True, exist_ok=True)

    model = MobileNetSegmentationNet(
        in_channels=cfg.num_input_channels,
        num_output_classes=cfg.num_output_classes,
        pretrained=cfg.pretrained_backbone,
        fpn_channels=cfg.fpn_channels,
        target_size=target_size,
    ).to(cfg.device)

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    backbone_frozen = True  # tracked explicitly so we don't re-trigger the freeze->unfreeze transition after resuming

    if cfg.resume and resume_path.exists():
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = load_resume_checkpoint(resume_path, cfg.device)

        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        epochs_without_improvement = ckpt["epochs_without_improvement"]

        # put the model into whatever freeze/unfreeze stage it was in at that epoch,
        # *before* building the optimizer, so the optimizer's parameter group matches
        # what was saved (this is the part that breaks if done in the wrong order)
        backbone_frozen = start_epoch < cfg.freeze_epochs
        if backbone_frozen:
            model.freeze_backbone()
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.lr_frozen, weight_decay=cfg.weight_decay,
            )
        else:
            model.unfreeze_backbone()
            optimizer = optim.Adam(model.parameters(), lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        restore_rng_state(ckpt)

        print(f"Resumed at epoch {start_epoch} | best_val_loss={best_val_loss:.5f} | "
              f"epochs_without_improvement={epochs_without_improvement} | "
              f"backbone_frozen={backbone_frozen}")
    else:
        if cfg.resume:
            print(f"--resume was set but no checkpoint found at {resume_path}; starting fresh.")
        # --- stage 1: freeze backbone ---
        model.freeze_backbone()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr_frozen, weight_decay=cfg.weight_decay,
        )

    try:
        for epoch in range(start_epoch, cfg.num_epochs):
            # --- stage 2: unfreeze at the configured epoch (only triggers once) ---
            if backbone_frozen and epoch >= cfg.freeze_epochs:
                print(f"[epoch {epoch}] unfreezing backbone, switching to lr_finetune={cfg.lr_finetune}")
                model.unfreeze_backbone()
                optimizer = optim.Adam(model.parameters(), lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)
                backbone_frozen = False

            model.train()
            if backbone_frozen:
                model.freeze_backbone()  # re-assert BatchNorm eval() each epoch (model.train() above flips it back)

            running_loss = 0.0
            for images, masks in train_loader:
                images = images.to(cfg.device)
                masks = masks.to(cfg.device)

                optimizer.zero_grad()
                logits = model(images)
                loss = segmentation_loss(logits, masks)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            n_batches = len(train_loader)
            stage = "frozen" if backbone_frozen else "finetune"
            train_loss = running_loss / n_batches

            val_metrics = evaluate(model, val_loader, cfg.device)

            print(f"[{stage}] epoch {epoch+1}/{cfg.num_epochs} | train_loss={train_loss:.5f}")
            print(f"           val_loss={val_metrics['val_loss']:.5f} | "
                  f"accuracy={val_metrics['accuracy']:.4f} | "
                  f"precision={val_metrics['precision']:.4f} | "
                  f"recall={val_metrics['recall']:.4f} | "
                  f"f1={val_metrics['f1']:.4f} | "
                  f"iou={val_metrics['iou']:.4f} | "
                  f"dice={val_metrics['dice']:.4f}")
            print(f"           cloud_pixel_fraction (val set)={val_metrics['cloud_pixel_fraction']:.4f}")

            # per-epoch visualization snapshot on the fixed val sample
            with torch.no_grad():
                viz_logits = model(viz_image.unsqueeze(0).to(cfg.device))
                viz_pred_probs = torch.sigmoid(viz_logits)[0, 0]  # (H, W)
            save_epoch_visualization(epoch, viz_image, viz_true_mask, viz_pred_probs, cfg.visualization_dir)

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                epochs_without_improvement = 0
                best_path = Path(cfg.checkpoint_dir) / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "config": asdict(cfg),
                }, best_path)
                print(f"           -> saved new best checkpoint to {best_path}")
            else:
                epochs_without_improvement += 1

            # save resume state EVERY epoch (not just on improvement) so an interruption
            # never loses more than the current in-progress epoch
            save_resume_checkpoint(resume_path, epoch, model, optimizer,
                                    best_val_loss, epochs_without_improvement, cfg)

            if epochs_without_improvement >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(no val improvement for {cfg.early_stopping_patience} epochs)")
                break

    except KeyboardInterrupt:
        print(f"\nTraining interrupted. Progress through the last completed epoch is saved at "
              f"{resume_path}. Re-run with --resume true to continue from there.")
        return

    print(f"Training finished. Best val_loss={best_val_loss:.5f}")


if __name__ == "__main__":
    config = parse_args()
    print("Config:")
    for k, v in asdict(config).items():
        print(f"  {k}: {v}")
    train(config)