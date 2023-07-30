import numpy as np


def window_image(
    image: np.ndarray, window_center: int, window_width: int
) -> np.ndarray:
    """Returns a transformed image centered a `window_center`. The resulting image range is:
    [`window_center` - `window_width` //2, `window_center` + `window_width` //2]"""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image
