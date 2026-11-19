import json
import os


def get_subject_features(features_dir, subject_dir):
    # MRI in/out phase: use the registered volumes if present, otherwise the originals
    reg_dir = os.path.join(subject_dir, "new_registration_output_final")
    reg_in  = os.path.join(reg_dir, "mri_in_phase_final.nii.gz")
    reg_out = os.path.join(reg_dir, "mri_out_phase_final.nii.gz")

    use_registered = (
        os.path.isdir(reg_dir)
        and os.path.isfile(reg_in)
        and os.path.isfile(reg_out)
    )

    if use_registered:
        mri_in_path  = reg_in
        mri_out_path = reg_out
    else:
        mri_in_path  = os.path.join(features_dir, "mri_combined_in_phase.nii.gz")
        mri_out_path = os.path.join(features_dir, "mri_combined_out_phase.nii.gz")

    paths = {
        "nacpet":                  os.path.join(features_dir, "nacpet.nii.gz"),
        "topogram":                os.path.join(features_dir, "topogram.nii.gz"),
        "mri_combined_in_phase":   mri_in_path,
        "mri_combined_out_phase":  mri_out_path,
        "mri_face_mask":           os.path.join(features_dir, "mri_face_mask.nii.gz"),
        **{f"mri_chunk_{i}_{phase}": os.path.join(features_dir, f"mri_chunk_{i}_{phase}.nii.gz")
           for i in range(4) for phase in ("in_phase", "out_phase")},
    }
    with open(os.path.join(features_dir, "metadata.json")) as f:
        metadata = json.load(f)

    # Min-max normalize demographics to [0, 1] (ranges from the training set)
    age    = float(metadata["age"])
    sex    = metadata["sex"]
    height = float(metadata["height"])
    weight = float(metadata["weight"])
    bmi    = weight / (height ** 2)

    metadata_norm = {
        "age_norm":    (age - 22) / (88 - 22),
        "sex_bin":     1.0 if sex == "M" else 0.0,
        "height_norm": (height - 1.54) / (1.945 - 1.54),
        "weight_norm": (weight - 54) / (132 - 54),
        "bmi_norm":    (bmi - 15.0) / (45.0 - 15.0),
    }

    return {**paths, **metadata, **metadata_norm, "_used_registered_mri": use_registered}


def get_subject_ct_labels(ct_label_dir):
    return {
        "ct":               os.path.join(ct_label_dir, "ct.nii.gz"),
        "body_seg":         os.path.join(ct_label_dir, "body_seg.nii.gz"),
        "organ_seg":        os.path.join(ct_label_dir, "organ_seg.nii.gz"),
        "prediction_mask":  os.path.join(ct_label_dir, "prediction_mask.nii.gz"),
    }


def get_dataset(data_dir):
    subjects = []
    for sub in sorted(os.listdir(data_dir)):
        subject_dir = os.path.join(data_dir, sub)
        subject = get_subject_features(
            os.path.join(subject_dir, "features"),
            subject_dir,
        )
        subject.update(get_subject_ct_labels(os.path.join(subject_dir, "ct-label")))
        subjects.append(subject)
    return subjects