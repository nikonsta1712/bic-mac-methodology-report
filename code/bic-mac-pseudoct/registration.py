"""
Automated MRI -> PET pitch registration for a SINGLE subject (container-ready).

Fully automatic + deterministic, so it reproduces on the hidden test subjects:
  * pitch estimated from head geometry (2D fit) + coarse-to-fine MI refinement
  * pitch estimated ONCE on in-phase, reused for out-phase (halves cost)
  * per-bed translations fixed and identical across subjects

SPEED: the MI angular search runs on a CROPPED head ROI (not the full
512x512x531 volume), which is the main cost driver. The final chosen angle is
then applied at FULL resolution so the stitched output is unchanged.

Writes corrected stitched MRI into OUT_SUBDIR next to features/, using the
filenames get_subject_features() looks for.

Standalone timing test (in your venv, isolates registration from inference):
    python registration.py --features_dir <...>/val/sub-004/features
"""

import os
import time
import numpy as np
import SimpleITK as sitk



# Constants (identical across all subjects)

SHIFTS = {
    1: (0.0, -15.0, -15.0),
    2: (0.0, -15.0, -10.0),
    3: (0.0, -10.0, -10.0),
}
POST_TRANSLATION_Y_MM = 0.0
POST_TRANSLATION_Z_MM = -10.0

HEAD_FRACTION   = 0.40
PET_THRESH_FRAC = 0.05

PITCH_COARSE_RANGE = (-20.0, 35.0)
PITCH_COARSE_STEP  = 2.5
PITCH_FINE_HALF    = 2.5
PITCH_FINE_STEP    = 0.25

# ROI margin (fraction of the head-mask extent per axis) added around the head before the MI sweep, so the head stays inside the crop after rotation

ROI_MARGIN_FRAC = 0.5

OUT_SUBDIR   = "new_registration_output_final"
OUT_TEMPLATE = "mri_{phase}_final.nii.gz"



# Pitch estimation (2D geometry)

def estimate_pitch_2d(head_mask_arr):
    pts = np.argwhere(head_mask_arr > 0)
    if len(pts) < 100:
        print("  [reg] Not enough voxels for 2D fit")
        return 0.0
    z = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    a, _ = np.polyfit(z, y, 1)
    angle_deg = np.rad2deg(np.arctan(a))
    print(f"  [reg] 2D pitch estimate: {angle_deg:+.2f} deg")
    return angle_deg



# Transform + MI

def _make_transform(pet_sitk, head_mask_arr, rot_x_deg, trans_y_mm=0.0, trans_z_mm=0.0):
    head_vox = np.argwhere(head_mask_arr > 0)
    com_vox = head_vox.mean(axis=0)
    com_phys = pet_sitk.TransformIndexToPhysicalPoint(
        [int(com_vox[2]), int(com_vox[1]), int(com_vox[0])]
    )
    tx = sitk.Euler3DTransform()
    tx.SetCenter(com_phys)
 
    # Resample applies the INVERSE transform (output->input): to rotate the content by +rot_x_deg we set the transform rotation to -rot_x_deg
 
    tx.SetRotation(float(np.deg2rad(-rot_x_deg)), 0.0, 0.0)
    tx.SetTranslation((0.0, float(trans_y_mm), float(trans_z_mm)))
    return tx


def compute_mi(fixed_sitk, moving_sitk):
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed=42)   # seeded -> reproducible
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(sitk.Euler3DTransform(), inPlace=False)
    return reg.MetricEvaluate(fixed_sitk, moving_sitk)



# Head ROI cropping (physical space preserved)
# numpy array axes are (z, y, x); sitk index axes are (x, y, z).

def _head_bbox(head_mask_arr, margin_frac=ROI_MARGIN_FRAC):
    pts = np.argwhere(head_mask_arr > 0)
    (zmin, ymin, xmin) = pts.min(axis=0)
    (zmax, ymax, xmax) = pts.max(axis=0)
    Z, Y, X = head_mask_arr.shape

    def pad(lo, hi, n):
        m = int(round(margin_frac * (hi - lo + 1))) + 5
        return max(0, int(lo) - m), min(n - 1, int(hi) + m)

    zmin, zmax = pad(zmin, zmax, Z)
    ymin, ymax = pad(ymin, ymax, Y)
    xmin, xmax = pad(xmin, xmax, X)
    return zmin, zmax, ymin, ymax, xmin, xmax


def _roi(img_sitk, bbox):
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    index = [int(xmin), int(ymin), int(zmin)]                       # sitk (x,y,z)
    size = [int(xmax - xmin + 1), int(ymax - ymin + 1), int(zmax - zmin + 1)]
    return sitk.RegionOfInterest(img_sitk, size, index)


def _mi_sweep(orig_full_sitk, pet_head_sitk, pet_sitk, head_mask_arr, angles):
    """Sweep angles on the cropped head ROI. Mattes MI is MINIMIZED (-MI)."""
    best_score = np.inf
    best_angle = float(angles[len(angles) // 2])
    for ang in angles:
        tx = _make_transform(pet_sitk, head_mask_arr, float(ang))
        moving_head = sitk.Resample(orig_full_sitk, pet_head_sitk, tx,
                                    sitk.sitkLinear, 0.0,
                                    orig_full_sitk.GetPixelID())
        score = compute_mi(pet_head_sitk, moving_head)
        if score < best_score:
            best_score = score
            best_angle = float(ang)
    return best_angle, best_score


def find_best_pitch_angle(original_arr, pet_sitk, pet_arr, head_mask_arr, init_angle):
    # Build the moving image once, and the cropped fixed head ROI once.
    orig_full = sitk.GetImageFromArray(original_arr.astype(np.float32))
    orig_full.CopyInformation(pet_sitk)

    pts = np.argwhere(head_mask_arr > 0)
    if len(pts) < 100:
        # degenerate: skip crop, sweep on full volume (rare / safety)
        pet_head = pet_sitk
    else:
        pet_head = _roi(pet_sitk, _head_bbox(head_mask_arr))
    print(f"  [reg] MI ROI size (x,y,z): {pet_head.GetSize()}  "
          f"(full was {pet_sitk.GetSize()})")

    coarse = np.arange(PITCH_COARSE_RANGE[0], PITCH_COARSE_RANGE[1] + 1e-6, PITCH_COARSE_STEP)
    coarse_angle, _ = _mi_sweep(orig_full, pet_head, pet_sitk, head_mask_arr, coarse)

    fine = np.arange(coarse_angle - PITCH_FINE_HALF,
                     coarse_angle + PITCH_FINE_HALF + 1e-6, PITCH_FINE_STEP)
    best_angle, _ = _mi_sweep(orig_full, pet_head, pet_sitk, head_mask_arr, fine)

    lo, hi = PITCH_COARSE_RANGE
    best_angle = float(np.clip(best_angle, lo, hi))
    print(f"  [reg] best pitch angle: {best_angle:+.2f} deg")
    return best_angle



# stitching

def compute_chunks_z(chunks_arr):
    chunks_z = {}
    for i, arr in chunks_arr.items():
        nonzero_z = np.where((arr > 10).sum(axis=(1, 2)) > 100)[0]
        chunks_z[i] = (int(nonzero_z.min()), int(nonzero_z.max())) if len(nonzero_z) else None
    return chunks_z


def compute_exclusive_zones(chunks_z):
    valid = {i: v for i, v in chunks_z.items() if v is not None}
    sorted_ids = sorted(valid.keys(), key=lambda i: valid[i][1], reverse=True)
    boundaries = {}
    for idx in range(len(sorted_ids) - 1):
        i, j = sorted_ids[idx], sorted_ids[idx + 1]
        boundaries[(i, j)] = int((valid[i][0] + valid[j][1]) / 2)
    exclusive = {}
    for idx, i in enumerate(sorted_ids):
        upper = valid[i][1] if idx == 0 else boundaries[(sorted_ids[idx - 1], i)]
        lower = 0 if idx == len(sorted_ids) - 1 else boundaries[(i, sorted_ids[idx + 1])]
        exclusive[i] = (lower, upper)
    return exclusive


def build_head_mask_from_pet(pet_array, chunks_z):
    z_min, z_max = chunks_z[0]
    head_z_end = int(z_min + HEAD_FRACTION * (z_max - z_min))
    slab = pet_array[z_min:head_z_end]
    thr = PET_THRESH_FRAC * slab.max()
    mask = np.zeros_like(pet_array, dtype=np.uint8)
    mask[z_min:head_z_end] = (slab > thr).astype(np.uint8)
    return mask, (z_min, head_z_end)


def apply_post_correction(original_arr, pet_sitk, head_mask_arr,
                          rot_x_deg, trans_y_mm, trans_z_mm):
    """Final FULL-resolution application of the chosen transform -> array."""
    tx = _make_transform(pet_sitk, head_mask_arr, rot_x_deg, trans_y_mm, trans_z_mm)
    orig = sitk.GetImageFromArray(original_arr.astype(np.float32))
    orig.CopyInformation(pet_sitk)
    corrected = sitk.Resample(orig, pet_sitk, tx, sitk.sitkLinear, 0.0,
                              orig.GetPixelID())
    return sitk.GetArrayFromImage(corrected)


def _load_chunks(features_dir, phase):
    chunks_sitk, chunks_arr = {}, {}
    for i in range(4):
        img = sitk.ReadImage(
            os.path.join(features_dir, f"mri_chunk_{i}_{phase}.nii.gz"),
            sitk.sitkFloat32)
        chunks_sitk[i] = img
        chunks_arr[i] = sitk.GetArrayFromImage(img)
    return chunks_sitk, chunks_arr


def _stitch_phase(features_dir, phase, pet, pet_arr, best_angle, head_mask_arr):
    chunks_sitk, chunks_arr = _load_chunks(features_dir, phase)
    exclusive_z = compute_exclusive_zones(compute_chunks_z(chunks_arr))

    registered = {
        0: apply_post_correction(chunks_arr[0], pet, head_mask_arr, best_angle,
                                 POST_TRANSLATION_Y_MM, POST_TRANSLATION_Z_MM)
    }
    for i in (1, 2, 3):
        tx = sitk.TranslationTransform(3)
        tx.SetOffset(SHIFTS[i])
        r = sitk.Resample(chunks_sitk[i], pet, tx, sitk.sitkLinear, 0.0,
                          chunks_sitk[i].GetPixelID())
        registered[i] = sitk.GetArrayFromImage(r)

    stitched = np.zeros_like(pet_arr)
    for z in range(pet_arr.shape[0]):
        for i, (z_min, z_max) in exclusive_z.items():
            if z_min <= z < z_max:
                stitched[z] = registered[i][z]
                break

    out = sitk.GetImageFromArray(stitched.astype(np.float32))
    out.CopyInformation(pet)
    return out



# Public entry point

def register_subject(features_dir, out_dir=None):
    t0 = time.time()
    pet = sitk.ReadImage(os.path.join(features_dir, "nacpet.nii.gz"), sitk.sitkFloat32)
    pet_arr = sitk.GetArrayFromImage(pet)

    _, chunks_arr_in = _load_chunks(features_dir, "in_phase")
    chunks_z = compute_chunks_z(chunks_arr_in)
    head_mask_arr, _ = build_head_mask_from_pet(pet_arr, chunks_z)

    # estimate pitch ONCE (in-phase), reuse for both phases
    init_angle = estimate_pitch_2d(head_mask_arr)
    t_pitch0 = time.time()
    best_angle = find_best_pitch_angle(chunks_arr_in[0], pet, pet_arr,
                                       head_mask_arr, init_angle)
    print(f"  [reg] pitch search took {time.time() - t_pitch0:.1f} s")

    if out_dir is None:
        subject_dir = os.path.dirname(os.path.normpath(features_dir))
        out_dir = os.path.join(subject_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    paths = {}
    for phase in ("in_phase", "out_phase"):
        out = _stitch_phase(features_dir, phase, pet, pet_arr, best_angle, head_mask_arr)
        p = os.path.join(out_dir, OUT_TEMPLATE.format(phase=phase))
        sitk.WriteImage(out, p)
        paths[phase] = p
        print(f"  [reg] wrote {p}")

    print(f"  [reg] TOTAL registration time: {time.time() - t0:.1f} s")
    return paths


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    args = ap.parse_args()
    register_subject(args.features_dir)
    print("DONE.")
