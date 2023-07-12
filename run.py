import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from absl import app, flags
from ml_collections import config_flags

from importlib import import_module

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    help_string="Path to the configuration file in configs/.",
    lock_config=False,
)
flags.mark_flags_as_required(["config"])


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    module = import_module(f"train_{FLAGS.config.model_name}")
    train_fn = getattr(module, "train")

    train_fn(FLAGS.config)


if __name__ == "__main__":
    app.run(main)
