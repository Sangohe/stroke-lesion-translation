import cv2
import numpy as np

from scipy import ndimage
from typing import Tuple, Callable

# Mask weights.
# ----------------------------------------------------------------


def dilated_sample_weights(
    mask: np.ndarray, inf_val: float = 0.1, sup_val: float = 0.8, steps: int = 64
) -> np.ndarray:
    """Computes sample weights for a binary mask by dilating it with a structuring element.

    Args:
        mask (np.ndarray): Binary mask to compute sample weights for.
        inf_val (float, optional): Minimum sample weight. Defaults to 0.1.
        sup_val (float, optional): Maximum sample weight. Defaults to 0.8.
        steps (int, optional): Number of steps between minimum and maximum sample weight. Defaults to 64.

    Returns:
        np.ndarray: Dilated sample weights for the binary mask slice.
    """
    dilated_weights = []
    for slice_idx in range(mask.shape[-1]):
        dilated_weights.append(
            _dilated_sample_weights(
                mask[..., slice_idx],
                inf_val=inf_val,
                sup_val=sup_val,
                steps=steps,
            )
        )
    dilated_weights = np.stack(dilated_weights, axis=-1)
    return dilated_weights


def _dilated_sample_weights(
    mask: np.ndarray, inf_val: float = 0.1, sup_val: float = 0.8, steps: int = 64
) -> np.ndarray:
    """Computes sample weights for a binary mask slice by dilating it with a structuring element.

    Args:
        mask (np.ndarray): Binary mask to compute sample weights for.
        inf_val (float, optional): Minimum sample weight. Defaults to 0.1.
        sup_val (float, optional): Maximum sample weight. Defaults to 0.8.
        steps (int, optional): Number of steps between minimum and maximum sample weight. Defaults to 64.

    Returns:
        np.ndarray: Dilated sample weights for the binary mask slice.
    """
    KERNEL_SIZE: int = 2

    mask_int = mask.astype(np.uint8)
    weights = np.linspace(sup_val, inf_val, steps)

    sample_weights = np.ones(mask.shape[:2]) * inf_val
    sample_weights[mask_int.astype(np.uint8) == 1] = sup_val

    structuring_elem = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * KERNEL_SIZE + 1, 2 * KERNEL_SIZE + 1),
        (KERNEL_SIZE, KERNEL_SIZE),
    )

    mask_to_dilate = mask_int
    for k in range(weights.shape[0] - 1):
        img_d = cv2.dilate(mask_to_dilate, structuring_elem)
        sample_weights[img_d - mask_to_dilate.astype(np.uint8) == 1] = weights[k + 1]
        mask_to_dilate = img_d

    return sample_weights


def sample_weights(mask: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    """Create a `sample_weights` array with the same dimensions of
    `mask`. The values of the `sample_weights` array will be determined
    by the `class_weights` array.

    Args:
        mask (np.ndarray): reference mask
        class_weights (np.ndarray): importance of each class

    Returns:
        np.ndarray: sample weights
    """
    sample_weights = np.take(class_weights, mask.astype(np.int64))
    return sample_weights


def label_uncertainty(mask: np.ndarray) -> np.ndarray:
    """Computes the label uncertainty weights proposed in:
    https://arxiv.org/abs/2102.04566

    Args:
        mask (np.ndarray): reference mask

    Returns:
        np.ndarray: uncertainty weights
    """
    mask = mask.astype(np.float32)
    std = mask.std()
    exp_num = ndimage.distance_transform_edt(mask) ** 2
    exp_den = 2 * (std**2)
    exp = np.exp(-(exp_num / exp_den))

    return 1 - exp


# Normalizations.
# ----------------------------------------------------------------


def min_max_normalization(data: np.ndarray) -> np.ndarray:
    """
    Normalize the given numpy array using min-max normalization.

    Parameters:
    data (np.ndarray): The input numpy array to be normalized.

    Returns:
    np.ndarray: The normalized numpy array.
    """
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)


def z_normalization(arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Normalize the given numpy array using z-score normalization.

    Parameters:
    arr (np.ndarray): The input numpy array to be normalized.

    Returns:
    Tuple[np.ndarray, float, float]: A tuple containing the normalized
    numpy array, mean, and standard deviation.
    """
    mean = arr.mean()
    std = max(arr.std(), 1e-8)
    arr_norm = (arr - mean) / std
    return arr_norm, mean, std


def z_denormalization(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Denormalize the given numpy array using z-score normalization.

    Parameters:
    arr (np.ndarray): The input numpy array to be denormalized.
    mean (float): The mean used for normalization.
    std (float): The standard deviation used for normalization.

    Returns:
    np.ndarray: The denormalized numpy array.
    """
    arr_denorm = arr * std + mean
    return arr_denorm


def ct_normalization(arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Normalize the given numpy array using contrast-limited adaptive
    histogram equalization (CLAHE).

    Parameters:
    arr (np.ndarray): The input numpy array to be normalized.

    Returns:
    Tuple[np.ndarray, float, float]: A tuple containing the normalized
    numpy array, mean, and standard deviation.
    """
    mean = arr.mean()
    std = max(arr.std(), 1e-8)
    lower_bound = np.percentile(arr, 0.05)
    upper_bound = np.percentile(arr, 99.5)
    arr = np.clip(arr, lower_bound, upper_bound)
    arr_norm = (arr - lower_bound) / (upper_bound - lower_bound)
    return arr_norm, mean, std


# Utilities.
# ----------------------------------------------------------------


def get_idxs_of_annotated_slices(mask: np.ndarray) -> np.ndarray:
    """Return an array of booleans that indicate which slices have
    lesions. Ideal for numpy fancy indexing.

    Args:
        mask (np.ndarray): mask with annotations.

    Returns:
        np.ndarray: Array indicating which slices have lesions.
    """
    if mask.shape[0] == mask.shape[1]:
        mask = mask.transpose(2, 0, 1)
    return np.array([True if np.count_nonzero(s) > 0 else False for s in mask])


def assert_mask_integrity(mask: np.ndarray):
    """Use this function in other functions to verify that the mask
    values are either zeros or zeros and ones.

    Args:
        mask (np.ndarray): mask to verify
    """
    assert mask.min() == 0.0
    assert mask.max() <= 1.0
    assert np.unique(mask).shape[0] <= 2
