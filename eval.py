import numpy as np
import nibabel as nib
from tqdm import tqdm
import tensorflow as tf

import shutil
from pathlib import Path
from typing import List, Union, Tuple

from preprocessing import min_max_normalization, window_image


def predict_and_save(
    generator: tf.keras.Model,
    patients_dirs: List[Union[str, Path]],
    experiment_dir: Union[str, Path],
    ones_norm: bool = False,
    resize_shape: Tuple[int, int] = (128, 128),
):
    if isinstance(experiment_dir, str):
        experiment_dir = Path(experiment_dir)

    metrics = ["psnr", "ssim"]
    metrics_dict = {"patient_id": [], "slice_index": [], "num_lesion_pixels": []}
    metrics_dict = {**metrics_dict, **{metric: [] for metric in metrics}}
    metrics_dict = {
        **metrics_dict,
        **{f"{metric}_lesion": [] for metric in metrics},
    }
    for patient_dir in tqdm(patients_dirs):
        patient_dir = Path(patient_dir)

        patient_id = patient_dir.name
        adc_path = patient_dir / f"{patient_id}_adc.nii.gz"
        ncct_path = patient_dir / f"{patient_id}_ncct.nii.gz"
        mask_path = patient_dir / f"masks/{patient_id}_r1_mask.nii.gz"

        ncct = nib.load(ncct_path)
        ncct_arr = ncct.get_fdata()
        ncct_arr_win = window_image(ncct_arr, 40, 80)
        ncct_min, ncct_max = np.min(ncct_arr), np.max(ncct_arr)
        ncct_arr = min_max_normalization(ncct_arr).astype(np.float32)
        ncct_arr_res = tf.image.resize(ncct_arr, resize_shape).numpy()
        # ncct_arr_win = min_max_normalization(ncct_arr_win)
        # ncct_arr_win_res = tf.image.resize(ncct_arr_win, resize_shape).numpy()

        adc = nib.load(adc_path)
        adc_arr = adc.get_fdata()
        adc_min, adc_max = np.min(adc_arr), np.max(adc_arr)
        adc_arr = min_max_normalization(adc_arr).astype(np.float32)
        adc_arr_res = tf.image.resize(adc_arr, resize_shape).numpy()

        # Compute the fake image.
        ncct_arr_res = (ncct_arr_res - 0.5) * 2.0 if ones_norm else ncct_arr_res
        adc_fake_arr_res = generator.predict(
            ncct_arr_res.transpose(2, 0, 1)[..., np.newaxis], verbose=0
        )
        adc_fake_arr_res = adc_fake_arr_res.transpose(1, 2, 0, 3)[..., 0]
        adc_fake_arr_res = (
            (adc_fake_arr_res / 2.0) + 0.5 if ones_norm else adc_fake_arr_res
        )

        # Reshape back to original shape.
        adc_fake_arr = tf.image.resize(adc_fake_arr_res, adc_arr.shape[:2]).numpy()

        # nifties.
        save_dir = experiment_dir / "predictions/test" / patient_id
        save_dir.mkdir(parents=True, exist_ok=True)
        adc_fake_arr_back = min_max_back_normalization(adc_fake_arr, adc_min, adc_max)
        adc_fake = nib.Nifti1Image(
            adc_fake_arr_back.astype(np.float32), adc.affine, adc.header
        )
        nib.save(adc_fake, save_dir / f"{patient_id}_adc_fake.nii.gz")
        shutil.copy(ncct_path, save_dir / f"{patient_id}_ncct.nii.gz")
        shutil.copy(adc_path, save_dir / f"{patient_id}_adc.nii.gz")

        # Compute metrics.
        if mask_path.exists():
            shutil.copy(mask_path, save_dir / f"{patient_id}_r1_mask.nii.gz")
            mask = nib.load(mask_path)
            mask_arr = mask.get_fdata()
            assert mask_arr.shape == adc_arr.shape


def min_max_back_normalization(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    return (x * (x_max - x_min)) + x_min
