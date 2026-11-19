from monai.transforms import *
from monai.transforms import MapTransform


class BroadcastTopogramd(MapTransform):
    """
    Repeat the 2D topogram along its singleton axis so that it matches the
    shape of the 3D reference image (e.g. NAC-PET), turning it into a full
    3D input channel. Topogram and NAC-PET/MRI share the same grid, so the
    broadcast is spatially aligned.
    """
    def __init__(self, keys, ref_key):
        super().__init__(keys)
        self.ref_key = ref_key

    def __call__(self, data):
        d = dict(data)
        ref = d[self.ref_key]                       # e.g. nacpet: (1, 512, 512, 531)
        for key in self.key_iterator(d):
            img = d[key]                            # topogram: (1, 512, 1, 531)
            reps = [1] * img.ndim
            for ax in range(1, img.ndim):           # skip channel axis 0
                if img.shape[ax] == 1 and ref.shape[ax] != 1:
                    reps[ax] = ref.shape[ax]
            out = img.repeat(*reps) if any(r > 1 for r in reps) else img
            # copy the reference affine (same grid) to avoid affine
            # mismatch warnings in ConcatItemsd
            if hasattr(out, "affine") and hasattr(ref, "affine"):
                out.affine = ref.affine.clone()
            d[key] = out
        return d


def get_cache_transforms():
    """Deterministic part (load + normalize + broadcast) -> safe to cache."""
    return Compose([
        LoadImaged(keys=["nacpet", "mri_combined_in_phase", "mri_combined_out_phase",
                         "topogram", "ct", "prediction_mask"]),
        EnsureChannelFirstd(keys=["nacpet", "mri_combined_in_phase", "mri_combined_out_phase",
                                  "topogram", "ct", "prediction_mask"]),

        NormalizeIntensityd(keys=["nacpet"], nonzero=False, channel_wise=True, subtrahend=[0]),
        NormalizeIntensityd(keys=["mri_combined_in_phase", "mri_combined_out_phase"], nonzero=False,
                            channel_wise=True),
        NormalizeIntensityd(keys=["topogram"], nonzero=False, channel_wise=True),

        ScaleIntensityRanged(keys=["ct"], a_min=-1000, a_max=2000, b_min=0.0, b_max=1.0, clip=True),

        BroadcastTopogramd(keys=["topogram"], ref_key="nacpet"),
    ])


def get_random_transforms(patch_size, num_samples=2):
    """Random augmentation + crop -> never cache this; it must run fresh every epoch."""
    return Compose([
        RandFlipd(
            keys=["nacpet", "mri_combined_in_phase", "mri_combined_out_phase",
                  "topogram", "ct", "prediction_mask"],
            spatial_axis=[0, 1, 2],
            prob=0.5,
        ),
        RandRotate90d(
            keys=["nacpet", "mri_combined_in_phase", "mri_combined_out_phase",
                  "topogram", "ct", "prediction_mask"],
            prob=0.3,
            max_k=3,
            spatial_axes=(0, 1),
        ),
        RandGaussianNoised(keys=["nacpet"], prob=0.3, mean=0.0, std=0.05),

        ConcatItemsd(
            keys=["nacpet", "mri_combined_in_phase", "mri_combined_out_phase", "topogram"],
            name="input"
        ),

        RandCropByPosNegLabeld(
            keys=["input", "ct", "prediction_mask"],
            label_key="prediction_mask",
            spatial_size=patch_size,
            pos=3,
            neg=1,
            num_samples=num_samples,
            image_key="input",
            image_threshold=0.0,
        ),

        EnsureTyped(
            keys=["input", "ct", "prediction_mask",
                  "age_norm", "sex_bin", "height_norm", "weight_norm", "bmi_norm"],
            dtype="float32",
        ),
    ])