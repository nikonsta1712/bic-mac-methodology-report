# Built on the official BIC-MAC baseline image. No network access is needed at inference.

FROM ghcr.io/bic-mac-challenge/baseline:latest

# Extra dependency for the MRI -> PET registration step.
RUN pip install --no-cache-dir --break-system-packages SimpleITK

WORKDIR /app

# Inference code
COPY model.py transforms.py dataset.py registration.py predict.py /app/

# Trained weights (predict.py looks for <script dir>/outputs/checkpoints/best_model.pth)
COPY best_model.pth /app/outputs/checkpoints/best_model.pth

# Challenge contract: read /data/features (read-only), write /data/output/ct.nii.gz
ENTRYPOINT ["python", "predict.py", \
            "--features_dir", "/data/features", \
            "--output_ct", "/data/output/ct.nii.gz"]
