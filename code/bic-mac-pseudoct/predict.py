import os
import argparse
import tempfile

import torch
import nibabel as nib
from pathlib import Path

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ConcatItemsd,
    EnsureTyped,
)

from model import build_model
from dataset import get_subject_features
from transforms import BroadcastTopogramd
from registration import register_subject          # automated MRI->PET pitch registration

# Inference settings
PATCH_SIZE = (192, 192, 192)
SW_BATCH = 2
OVERLAP = 0.50


def predict(features_dir, out_path):
    """
    Multimodal prediction using NAC-PET, MRI (in/out-phase) and topogram
    to generate a pseudo-CT.
    """

    # 1. Transforms — normalization + stacking with training.
    transforms = Compose([
        LoadImaged(keys=["nacpet", "mri_combined_in_phase", "mri_combined_out_phase", "topogram"]),
        EnsureChannelFirstd(keys=["nacpet", "mri_combined_in_phase", "mri_combined_out_phase", "topogram"]),

        NormalizeIntensityd(keys=["nacpet"], nonzero=False, subtrahend=[0], channel_wise=True),
        NormalizeIntensityd(keys=["mri_combined_in_phase", "mri_combined_out_phase"],
                            nonzero=False, channel_wise=True),
        NormalizeIntensityd(keys=["topogram"], nonzero=False, channel_wise=True),

        BroadcastTopogramd(keys=["topogram"], ref_key="nacpet"),

        ConcatItemsd(keys=["nacpet", "mri_combined_in_phase",
                           "mri_combined_out_phase", "topogram"], name="input"),
        EnsureTyped(keys=["input"], dtype="float32"),
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Build and load model (4 input channels)
    model = build_model().to(device)
    model_path = Path(__file__).parent / "outputs/checkpoints/best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train first!")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 3. Prepare subject paths
    subject_dir = os.path.dirname(os.path.normpath(features_dir))
    subject_features = get_subject_features(features_dir, subject_dir)

    # Run registration on-the-fly (test subjects have NO pre-registered files)
    # Write to a guaranteed-writable temp dir and point the MRI keys there explicitly.
    reg_out_dir = os.path.join(tempfile.gettempdir(), "bicmac_reg")
    reg_paths = register_subject(features_dir, out_dir=reg_out_dir)
    subject_features["mri_combined_in_phase"] = reg_paths["in_phase"]
    subject_features["mri_combined_out_phase"] = reg_paths["out_phase"]
    print(f"Registration done -> {reg_out_dir}")

    print("Loading and preprocessing multimodal images...")
    data = transforms(subject_features)

    # Add batch dim: [C, H, W, D] -> [1, C, H, W, D]
    x = data["input"].unsqueeze(0).to(device)

    # Metadata vector (pre-normalized in new_dataset.py)
    meta = torch.tensor([[
        float(subject_features["age_norm"]),
        float(subject_features["sex_bin"]),
        float(subject_features["height_norm"]),
        float(subject_features["weight_norm"]),
        float(subject_features["bmi_norm"]),
    ]], dtype=torch.float32).to(device)  # (1, 5)

    # 4. Sliding-window inference (metadata passed to every patch forward pass)
    print("Running sliding window inference...")
    with torch.no_grad():
        pred = sliding_window_inference(
            x,
            roi_size=PATCH_SIZE,
            sw_batch_size=SW_BATCH,
            predictor=lambda patch: model(patch, metadata=meta),
            overlap=OVERLAP,
            mode="gaussian",
            progress=True,
        )

    # 5. Post-process and save. Normalized [0,1] -> HU [-1000, 2000].
    pred_hu = pred.cpu().numpy()[0, 0] * 3000 - 1000

    # Same affine/grid as nacpet (required by the submission spec).
    affine = data["nacpet"].meta["affine"].numpy()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print("Saving pseudo-CT...")
    nib.save(nib.Nifti1Image(pred_hu.astype("float32"), affine), out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", required=True, help="Path to subject's features/ folder")
    parser.add_argument("--output_ct", required=True, help="Output path for pseudo-CT")
    args = parser.parse_args()

    predict(args.features_dir, args.output_ct)
