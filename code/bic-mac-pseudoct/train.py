import os
import random

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import yaml
from tqdm import tqdm

from monai.data import DataLoader, Dataset, CacheDataset
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ScaleIntensityRanged,
    ConcatItemsd,
    EnsureTyped,
)

from dataset import get_dataset
from transforms import get_cache_transforms, get_random_transforms, BroadcastTopogramd
from model import build_model


# ------------------------------------------------------------------
# Normalization / mu-space constants
# ------------------------------------------------------------------
HU_MIN = -1000.0
HU_RANGE = 3000.0          # 2000 - (-1000)

BONE_HU_THRESH = 150.0     # bone threshold (HU) for weighting
W_BONE = 5.0
LAMBDA_GRAD = 0.1


def load_config():
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# HU -> mu (linear attenuation coefficient @ 511 keV), differentiable
# ------------------------------------------------------------------
def hu_to_mu_torch(hu):
    """
    Bilinear HU -> mu (Carney et al. 2006, Med Phys 33:976-983) @ 511 keV.
    same as src/evaluation/metrics/ct_whole_body_mae.py::hu_to_mu.
      hu1000 = HU + 1000 ; breakpoint bp = 1047
      hu1000 <  bp :  mu = 9.6e-5 * hu1000
      hu1000 >= bp :  mu = 5.10e-5 * hu1000 + 4.71e-2
    """
    a, b, bp = 5.10e-5, 4.71e-2, 1047.0
    hu1000 = hu + 1000.0
    mu = torch.where(
        hu1000 < bp,
        9.6e-5 * hu1000,
        a * hu1000 + b,
    )
    return mu.clamp(min=0.0)


def norm_to_hu(x):
    """Normalized [0,1] -> HU."""
    return x * HU_RANGE + HU_MIN


def _spatial_gradient(x):
    gx = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
    gy = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
    gz = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
    return gx, gy, gz


def gradient_loss_mu(pred_mu, target_mu):
    """Spatial-gradient L1 in mu-space (bone/air interfaces)."""
    p_x, p_y, p_z = _spatial_gradient(pred_mu)
    t_x, t_y, t_z = _spatial_gradient(target_mu)
    return (torch.mean(torch.abs(p_x - t_x)) +
            torch.mean(torch.abs(p_y - t_y)) +
            torch.mean(torch.abs(p_z - t_z)))


def compute_loss(pred, y, mask):
    """
    Combined loss in mu-space:
      - bone-weighted, masked L1 (mu)
      - gradient (edge) L1 (mu)
    pred, y: normalized [0,1] tensors (B,1,H,W,D)
    mask:    bool prediction_mask (B,1,H,W,D)
    """
    pred_hu = norm_to_hu(pred)
    y_hu = norm_to_hu(y)

    pred_mu = hu_to_mu_torch(pred_hu)
    y_mu = hu_to_mu_torch(y_hu)

    mask_f = mask.float()

    # Bone voxels: from ground-truth CT, inside the mask
    bone = ((y_hu > BONE_HU_THRESH) & mask).float()

    # Weight map: 1 soft tissue, W_BONE bones, 0 outside the mask
    weight = mask_f * (1.0 + (W_BONE - 1.0) * bone)

    abs_err = torch.abs(pred_mu - y_mu)
    l1_mu = (weight * abs_err).sum() / weight.sum().clamp(min=1.0)

    grad_mu = gradient_loss_mu(pred_mu, y_mu)

    total = l1_mu + LAMBDA_GRAD * grad_mu
    return total, l1_mu.detach(), grad_mu.detach()


@torch.no_grad()
def compute_val_mu_mae(pred, y, mask):
    """
    (unweighted) whole-body mu-MAE inside mask
    -> directly comparable with mu_map_mae evaluation.
    """
    pred_mu = hu_to_mu_torch(norm_to_hu(pred))
    y_mu = hu_to_mu_torch(norm_to_hu(y))
    mask_f = mask.float()
    mae = (mask_f * torch.abs(pred_mu - y_mu)).sum() / mask_f.sum().clamp(min=1.0)
    return mae


def extract_metadata(batch, device):
    """(B, 5): age_norm, sex_bin, height_norm, weight_norm, bmi_norm."""
    meta = torch.stack([
        batch["age_norm"].float(),
        batch["sex_bin"].float(),
        batch["height_norm"].float(),
        batch["weight_norm"].float(),
        batch["bmi_norm"].float(),
    ], dim=1).to(device)
    return meta


# ------------------------------------------------------------------
# Whole-body validation transform
#   Same normalization as get_transforms (z-score),
#   Without augmentations, without crop -> whole volume.
# ------------------------------------------------------------------
def get_fullbody_val_transforms():
    return Compose([
        LoadImaged(keys=["nacpet", "mri_combined_in_phase", "mri_combined_out_phase",
                         "topogram", "ct", "prediction_mask"]),
        EnsureChannelFirstd(keys=["nacpet", "mri_combined_in_phase", "mri_combined_out_phase",
                                  "topogram", "ct", "prediction_mask"]),

        NormalizeIntensityd(keys=["nacpet"], nonzero=False, channel_wise=True, subtrahend=[0]),
        NormalizeIntensityd(keys=["mri_combined_in_phase", "mri_combined_out_phase"],
                            nonzero=False, channel_wise=True),
        NormalizeIntensityd(keys=["topogram"], nonzero=False, channel_wise=True),
        ScaleIntensityRanged(keys=["ct"], a_min=-1000, a_max=2000,
                             b_min=0.0, b_max=1.0, clip=True),

        BroadcastTopogramd(keys=["topogram"], ref_key="nacpet"),
        ConcatItemsd(keys=["nacpet", "mri_combined_in_phase",
                           "mri_combined_out_phase", "topogram"], name="input"),

        EnsureTyped(keys=["input", "ct", "prediction_mask",
                          "age_norm", "sex_bin", "height_norm",
                          "weight_norm", "bmi_norm"], dtype="float32"),
    ])


def main():
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ----------------------------------------------------------
    # 1. Data preparation  ->  8 held-out val subjects, unbiased split
    # ----------------------------------------------------------
    all_data = get_dataset(cfg["data_dir"])          # 75 subjects (sorted)

    split_seed = cfg.get("split_seed", 42)
    random.seed(split_seed)
    random.shuffle(all_data)

    n_val = cfg.get("n_val_subjects", 8)             # 8 val / 67 train
    val_data, train_data = all_data[:n_val], all_data[n_val:]
    print(f"Split: {len(train_data)} train / {len(val_data)} val (seed={split_seed})")

    def _sid(subj):
        for k in ("nacpet", "ct"):
            if k in subj:
                return os.path.normpath(subj[k]).split(os.sep)[-3]
        return "?"

    val_ids = [_sid(s) for s in val_data]
    print("Validation subjects:", val_ids)

    # ----------------------------------------------------------
    # Datasets: cache the deterministic preprocessing (load + normalize)
    # in RAM, apply random augmentation + crop fresh every epoch.
    # ----------------------------------------------------------
    num_workers = cfg.get("num_workers", 8)

    cache_transforms = get_cache_transforms()
    random_transforms = get_random_transforms(cfg["patch_size"], cfg.get("train_num_samples", 1))
    val_transforms = get_fullbody_val_transforms()

    print("Creating datasets (caching in RAM, first run will take longer)...")
    train_cache = CacheDataset(
        data=train_data,
        transform=cache_transforms,
        cache_rate=1.0,
        num_workers=num_workers,
    )
    train_dataset = Dataset(data=train_cache, transform=random_transforms)

    val_dataset = CacheDataset(
        data=val_data,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=num_workers,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )

    # ----------------------------------------------------------
    # 2. Model, optimizer
    # ----------------------------------------------------------
    model = build_model().to(device)                 # 4 input channels
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("learning_rate", 1e-4),
        weight_decay=1e-5,
    )

    out = cfg.get("output_dir", "outputs")
    os.makedirs(f"{out}/checkpoints", exist_ok=True)

    # ----------------------------------------------------------
    # 3. Validation settings (whole-body, every N epochs)
    # ----------------------------------------------------------
    epochs = cfg.get("epochs", 3)
    VAL_EVERY = cfg.get("val_every", 1)
    VAL_ROI = tuple(cfg.get("val_roi", (192, 192, 192)))       # same ROI as predict.py
    VAL_OVERLAP = cfg.get("val_overlap", 0.5)

    best_val_mae = float("inf")
    print("\nStarting training...\n")

    start_epoch = 0
    resume_path = f"{out}/checkpoints/last_state.pth"
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        best_val_mae = ckpt["best_val_mae"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch} (best val mu-MAE so far: {best_val_mae:.6f})")

    for epoch in range(start_epoch, epochs):
        # ---------------- Train ----------------
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

        for batch in pbar:
            x = batch["input"].to(device)
            y = batch["ct"].to(device)
            mask = batch["prediction_mask"].bool().to(device)
            meta = extract_metadata(batch, device)

            optimizer.zero_grad()
            pred = model(x, metadata=meta)

            total_loss, l1_mu, grad_mu = compute_loss(pred, y, mask)
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            pbar.set_postfix(loss=f"{total_loss.item():.5f}",
                             l1_mu=f"{l1_mu.item():.5f}")

        avg_train_loss = epoch_loss / len(train_loader)

        # ---------------- Whole-body validation ----------------
        is_val_epoch = ((epoch + 1) % VAL_EVERY == 0) or (epoch == epochs - 1)

        if is_val_epoch:
            model.eval()
            val_mae = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val - whole body]"):
                    x = batch["input"].to(device)
                    y = batch["ct"].to(device)
                    mask = batch["prediction_mask"].bool().to(device)
                    meta = extract_metadata(batch, device)

                    pred = sliding_window_inference(
                        x,
                        roi_size=VAL_ROI,
                        sw_batch_size=1,
                        predictor=lambda p: model(p, metadata=meta),
                        overlap=VAL_OVERLAP,
                        mode="gaussian",
                    )
                    val_mae += compute_val_mu_mae(pred, y, mask).item()

            avg_val_mae = val_mae / len(val_loader)
            print(f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.5f} "
                  f"| Whole-body Val mu-MAE: {avg_val_mae:.6f}")

            # Save best model (min whole-body mu-MAE)
            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
                torch.save(model.state_dict(), f"{out}/checkpoints/best_model.pth")
                print(f"   -> Best model saved (Val mu-MAE = {avg_val_mae:.6f})")
        else:
            print(f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.5f} | (val skipped)")

        # Full state -> resume-able (saved EVERY epoch, not only non-val ones)
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_mae": best_val_mae,
        }, f"{out}/checkpoints/last_state.pth")

    print(f"\nTraining completed! Best whole-body Val mu-MAE: {best_val_mae:.6f}")


if __name__ == "__main__":
    main()
