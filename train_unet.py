import ml_collections
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import os
from time import time

from dataloader import get_dataset
from schedulers import WarmUpCosine
from models.unet import UNet
from run_management import Logger, create_run_dir


def train(config: ml_collections.ConfigDict):
    tf.keras.utils.set_random_seed(config.random_seed)

    # Set up the directories needed for this run.
    experiment_desc = create_experiment_desc(config)
    run_dir = create_run_dir(experiment_desc, root_dir=config.root_dir)
    logs_fname = os.path.join(run_dir, "execution_logs.txt")
    _ = Logger(file_name=logs_fname, file_mode="a", should_flush=True)

    config.logs_dir = create_directory(os.path.join(run_dir, "logs"))
    config.weights_dir = create_directory(os.path.join(run_dir, "weights"))
    config.evaluation_dir = create_directory(os.path.join(run_dir, "evaluation"))
    print(f"Logs directory: {os.path.basename(config.logs_dir)}")
    print(f"Weights directory: {os.path.basename(config.weights_dir)}")
    print(f"Evaluation directory: {os.path.basename(config.evaluation_dir)}")

    print("Loading datasets...")
    if config.dataloader.class_weights:
        print(f"Dataloader will include {config.dataloader.weights_key}.")
    train_dset = get_dataset(config.dataloader.train_tfrecord_path, **config.dataloader)
    valid_dset = get_dataset(
        config.dataloader.valid_tfrecord_path,
        batch_size=config.dataloader.batch_size,
        class_weights=config.dataloader.class_weights,
    )

    print("Creating models and optimizers...")
    generator = UNet(
        input_shape=config.model.input_shape,
        init_dim=config.model.init_dim,
        dim=config.model.dim,
        dim_mults=config.model.dim_mults,
        groups=config.model.groups,
        out_channels=config.model.out_channels,
        out_activation=config.model.out_activation,
    )
    if config.use_scheduler:
        print("Using a cosine decay with warm-up scheduler.")
        NUM_TRAIN_SAMPLES = 7542  # ! Hardcoded for our preprocessed APIS dataset.
        total_steps = (
            int(NUM_TRAIN_SAMPLES / config.dataloader.batch_size) * config.epochs
        )
        warmup_steps = int(total_steps * config.warmup_epoch_percentage)
        lrs = WarmUpCosine(
            learning_rate_base=config.base_lr,
            total_steps=total_steps,
            warmup_learning_rate=config.warmup_lr,
            warmup_steps=warmup_steps,
        )
    else:
        print(f"Using a constant learning rate of {config.base_lr}")
        lrs = config.base_lr
    generator_optimizer = tf.keras.optimizers.Adam(lrs, beta_1=0.5)

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=config.logs_dir),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(
            config.weights_dir,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]
    generator.compile(generator_optimizer, loss=tf.keras.losses.MeanAbsoluteError())
    generator.fit(
        train_dset,
        epochs=config.epochs,
        callbacks=callbacks,
        validation_data=valid_dset,
        verbose=2,
    )

    print("Ending training...")


# Utilities.
# ------------------------------------------------


def create_experiment_desc(config: ml_collections.ConfigDict) -> str:
    args = [os.path.basename(config.dset_dir)]
    if config.dataloader.class_weights:
        args.append(config.dataloader.weights_key)
    args.extend(
        [
            "unet",
            config.optimizer_name,
            f"{config.dataloader.batch_size}bs"
            f"{config.base_lr:.0e}lr".replace("-", ""),
            f"{config.weight_decay:.0e}wd".replace("-", ""),
        ]
    )
    return "-".join(args).replace(".", "")


def create_directory(path: str) -> str:
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
    return path
