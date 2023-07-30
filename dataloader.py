import keras
import keras_cv
import tensorflow as tf

from typing import Tuple
from functools import partial

feature_desc = {
    "ncct": tf.io.FixedLenFeature([], tf.string),
    "ncct_shape": tf.io.FixedLenFeature([3], tf.int64),
    "adc": tf.io.FixedLenFeature([], tf.string),
    "adc_shape": tf.io.FixedLenFeature([3], tf.int64),
    "mask": tf.io.FixedLenFeature([], tf.string),
    "mask_shape": tf.io.FixedLenFeature([3], tf.int64),
    "weights": tf.io.FixedLenFeature([], tf.string),
    "weights_shape": tf.io.FixedLenFeature([3], tf.int64),
    "dilated_weights": tf.io.FixedLenFeature([], tf.string),
    "dilated_weights_shape": tf.io.FixedLenFeature([3], tf.int64),
    "volumetric_dilated_weights": tf.io.FixedLenFeature([], tf.string),
    "volumetric_dilated_weights_shape": tf.io.FixedLenFeature([3], tf.int64),
}


def get_dataset(
    tfrecord_path: str,
    augmentations: bool = False,
    class_weights: bool = False,
    weights_key: str = "dilated_weights",
    batch_size: int = 32,
    prefetch: bool = False,
    shuffle_size: int = 0,
    cache: bool = False,
    repeat: bool = False,
    **kwargs
) -> tf.data.Dataset:
    AUTOTUNE = tf.data.AUTOTUNE

    dset = tf.data.TFRecordDataset(tfrecord_path)
    dset = dset.map(
        partial(parse_example, weights_key=weights_key), num_parallel_calls=AUTOTUNE
    )
    dset = dset.cache() if cache else dset

    if augmentations:
        ROTATION_FACTOR = (-0.1, 0.1)
        augment = keras.Sequential(
            [
                keras_cv.layers.RandomFlip(),
                keras_cv.layers.RandomRotation(
                    factor=ROTATION_FACTOR,
                ),
                keras_cv.layers.RandomCutout((0.0, 0.15), (0.0, 0.15)),
            ]
        )

        def augmentation(
            ncct: tf.Tensor, adc: tf.Tensor, weights: tf.Tensor
        ) -> Tuple[tf.Tensor, tf.Tensor]:
            inputs_dict = {
                "images": tf.concat([ncct, adc, weights], axis=-1),
            }
            ncct, adc, weights = tf.split(
                augment(inputs_dict, training=True)["images"], 3, -1
            )
            return ncct, adc, weights

        dset = dset.map(augmentation, num_parallel_calls=AUTOTUNE)

    dset = dset.map(lambda x, y, w: (x, y)) if not class_weights else dset
    dset = dset.repeat() if repeat else dset
    dset = (
        dset.shuffle(shuffle_size, reshuffle_each_iteration=True)
        if shuffle_size
        else dset
    )
    dset = dset.batch(batch_size, num_parallel_calls=AUTOTUNE)
    dset = dset.prefetch(AUTOTUNE) if prefetch else dset
    return dset


def parse_example(example_proto, weights_key):
    example = tf.io.parse_single_example(example_proto, feature_desc)
    ncct = tf.io.parse_tensor(example["ncct"], tf.float32)
    ncct = tf.reshape(ncct, example["ncct_shape"])
    ncct = tf.image.resize(ncct, [256, 256])
    adc = tf.io.parse_tensor(example["adc"], tf.float32)
    adc = tf.reshape(adc, example["adc_shape"])
    adc = tf.image.resize(adc, [256, 256])
    weights = tf.io.parse_tensor(example[weights_key], tf.float32)
    weights = tf.reshape(weights, example[weights_key + "_shape"])
    interpolation = "nearest" if weights_key == "weights" else "bilinear"
    weights = tf.image.resize(weights, [256, 256], method=interpolation)
    return ncct, adc, weights
