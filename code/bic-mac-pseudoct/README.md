# BIC-MAC 2026 — Pseudo-CT synthesis from PET/MRI

Inference and training code for our submission to the
[BIC-MAC challenge](https://github.com/bic-mac-challenge) (MICCAI 2026):
whole-body pseudo-CT synthesis from non-attenuation-corrected PET, Dixon MRI and
the CT topogram, for PET attenuation correction.

Docker image: `nikonsta/bic-mac:latest`

## Method overview

1. **Automated MRI → PET registration** (`registration.py`)
   The per-bed Dixon MRI chunks are stitched and rigidly corrected into PET space.
   - Head mask from the NAC-PET (no dependency on `prediction_mask` at inference)
   - Head pitch: initial estimate from a 2D geometric fit, then a coarse-to-fine
     search (−20° to 35°, 2.5° → 0.25°) that minimizes the Mattes mutual-information
     metric between PET and MRI on a cropped head ROI (`sitk.Euler3DTransform`)
   - Pitch is estimated once on in-phase and reused for out-phase
   - Fixed per-bed translations and exclusive-zone stitching (identical for all subjects)
   - Runtime: about 4 min/subject on CPU

2. **Network** (`model.py`): 3D U-Net with residual blocks and attention gates
   (16–256 feature channels, 5.8M parameters).
   - Input: 4 channels: NAC-PET, MRI in-phase, MRI out-phase, topogram (broadcast to 3D)
   - Demographic conditioning: age, sex, height, weight and BMI (min-max normalized)
     are mapped through an MLP with a sigmoid output to channel-wise scaling factors
     applied at the bottleneck (FiLM-style, scale only)

3. **Loss** (`train.py`): computed in linear-attenuation (µ) space at 511 keV using
   the bilinear HU → µ mapping of Carney et al. (2006)
   - Bone-weighted masked L1 on µ (bone = GT > 150 HU, weight 5; zero outside
     `prediction_mask`)
   - plus 0.1 × spatial-gradient L1 on µ

4. **Inference** (`predict.py`): sliding window, 192³ patches, 0.5 overlap,
   Gaussian blending. Output is mapped from [0, 1] back to HU [−1000, 2000] and
   saved on the NAC-PET grid.

### Intensity normalization
| Input | Normalization |
|---|---|
| NAC-PET | divide by std (no mean subtraction) |
| MRI in/out-phase | per-volume z-score |
| Topogram | per-image z-score, broadcast to 3D |
| CT (target) | clip to [−1000, 2000] HU, scale to [0, 1] |

## Repository layout

```
predict.py        # inference entry point (registration + network)
registration.py   # automated MRI -> PET registration (also runnable standalone)
model.py          # Attention U-Net with demographic conditioning
dataset.py        # subject file discovery + demographic normalization
transforms.py     # MONAI transforms (topogram broadcast, augmentation, cropping)
train.py          # training script
config.yaml       # training configuration
best_model.pth    # trained weights
Dockerfile        # challenge submission container
```

## Inference

### Docker (as submitted)

```bash
docker build -t bic-mac .
docker run --gpus all \
  -v /path/to/sub-XXX/features:/data/features:ro \
  -v /path/to/output:/data/output \
  bic-mac
```

The container reads the challenge `features/` folder and writes `/data/output/ct.nii.gz`.

### Without Docker

```bash
pip install -r requirements.txt
mkdir -p outputs/checkpoints && cp best_model.pth outputs/checkpoints/
python predict.py --features_dir /path/to/sub-XXX/features --output_ct ./ct.nii.gz
```

Registration runs on the fly and writes intermediate files to the system temp
directory (`<tmp>/bicmac_reg`).

## Training

1. **Pre-register all training subjects** (registration is done offline for training):
   ```bash
   for s in /path/to/train/sub-*; do
       python registration.py --features_dir "$s/features"
   done
   ```
   This writes `new_registration_output_final/mri_{in,out}_phase_final.nii.gz` next to
   each `features/` folder; `dataset.py` picks these up automatically and falls back to
   the original MRI if they are missing.

2. Set `data_dir` in `config.yaml` and run:
   ```bash
   python train.py
   ```

Training setup used for the submitted model: 75 subjects (67 train / 8 held-out
validation, split seed 42), 192³ patches, 2 patches per subject per epoch sampled
3:1 foreground:background within `prediction_mask`, AdamW (lr 3e-4, weight decay
1e-5), 250 epochs, whole-body validation µ-MAE every 10 epochs (the best
checkpoint is kept). Augmentation: random flips, random 90° axial rotations,
Gaussian noise on the NAC-PET. Trained on a single cloud GPU. Training can be
resumed from `outputs/checkpoints/last_state.pth`.

## Citation

If you use this code, please cite the BIC-MAC challenge paper (in preparation).

## License

MIT. See `LICENSE`.
