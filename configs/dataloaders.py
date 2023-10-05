import os
import ml_collections


def get_dataloader_config(dset_dir: str) -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.train_tfrecord_path = os.path.join(dset_dir, "train.tfrecord")
    config.valid_tfrecord_path = os.path.join(dset_dir, "valid.tfrecord")
    config.test_patients = os.path.join(dset_dir, "test")
    config.augmentations = True
    config.class_weights = False
    config.weights_key = "weights"
    config.batch_size = 16
    config.prefetch = True
    config.shuffle_size = 1000
    config.cache = False
    config.repeat = False
    return config
