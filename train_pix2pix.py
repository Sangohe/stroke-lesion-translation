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
from models.pix2pix import Generator, Discriminator
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
    train_dset = get_dataset(config.dataloader.train_tfrecord_path, **config.dataloader)
    valid_dset = get_dataset(
        config.dataloader.valid_tfrecord_path,
        batch_size=config.dataloader.batch_size,
        class_weights=config.dataloader.class_weights,
    )

    print("Creating models and optimizers...")
    generator = Generator(config.model.out_channels, config.model.out_activation)
    discriminator = Discriminator()
    if config.use_scheduler:
        print("Using a cosine decay with warm-up scheduler.")
        NUM_TRAIN_SAMPLES = 7542  # ! Hardcoded for our preprocessed APIS dataset.
        total_steps = (
            int(NUM_TRAIN_SAMPLES / config.dataloader.batch_size) * config.epochs
        )
        print(total_steps)
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
    discriminator_optimizer = tf.keras.optimizers.Adam(lrs, beta_1=0.5)

    fit(
        config.weights_dir,
        config.logs_dir,
        train_dset,
        valid_dset,
        config.epochs,
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
    )

    print("Ending training...")


def fit(
    weights_dir: str,
    logs_dir: str,
    train_ds: tf.data.Dataset,
    valid_ds: tf.data.Dataset,
    epochs: int,
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    generator_optimizer: tf.keras.optimizers.Optimizer,
    discriminator_optimizer: tf.keras.optimizers.Optimizer,
):
    # ?: Need to find good example inputs.

    # Configure checkpoint manager.
    ckpt = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, weights_dir, max_to_keep=3)

    train_writer = tf.summary.create_file_writer(os.path.join(logs_dir, "train"))
    valid_writer = tf.summary.create_file_writer(os.path.join(logs_dir, "valid"))

    train_template = "Train [{}] -> gen_total_loss: {:.4f}, gen_gan_loss: {:.4f}, gen_l1_loss: {:.4f}, disc_loss: {:.4f}"
    valid_template = "Validation [{}] -> val_gen_total_loss: {:.4f}, val_gen_gan_loss: {:.4f}, val_gen_l1_loss: {:.4f}, val_disc_loss: {:.4f}"

    gen_total_loss_tracker = tf.keras.metrics.Mean(name="gen_total_loss")
    gen_gan_loss_tracker = tf.keras.metrics.Mean(name="gen_gan_loss")
    gen_l1_loss_tracker = tf.keras.metrics.Mean(name="gen_l1_loss")
    disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")

    val_gen_total_loss_tracker = tf.keras.metrics.Mean(name="val_gen_total_loss")
    val_gen_gan_loss_tracker = tf.keras.metrics.Mean(name="val_gen_gan_loss")
    val_l1_loss_tracker = tf.keras.metrics.Mean(name="val_l1_loss")
    val_disc_loss_tracker = tf.keras.metrics.Mean(name="val_disc_loss")
    best_val_gen_total_loss = tf.constant(np.inf)  # Keep track of best validation loss.

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")

        # Training.
        epoch_train_start = time()
        for step, batch in train_ds.enumerate():
            if len(batch) == 3:
                input_image, target, weights = batch
            else:
                input_image, target = batch
                weights = None

            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
                input_image,
                target,
                weights=weights,
            )

            gen_total_loss_tracker.update_state(gen_total_loss)
            gen_gan_loss_tracker.update_state(gen_gan_loss)
            gen_l1_loss_tracker.update_state(gen_l1_loss)
            disc_loss_tracker.update_state(disc_loss)

        train_duration = time() - epoch_train_start
        train_duration = f"{train_duration//60:.0f}m {train_duration%60:.0f}s"
        print(
            train_template.format(
                train_duration,
                gen_total_loss_tracker.result(),
                gen_gan_loss_tracker.result(),
                gen_l1_loss_tracker.result(),
                disc_loss_tracker.result(),
            )
        )

        with train_writer.as_default():
            tf.summary.scalar("lr", generator_optimizer.lr, step=epoch)
            tf.summary.scalar(
                "gen_total_loss", gen_total_loss_tracker.result(), step=epoch
            )
            tf.summary.scalar("gen_gan_loss", gen_gan_loss_tracker.result(), step=epoch)
            tf.summary.scalar("gen_l1_loss", gen_l1_loss_tracker.result(), step=epoch)
            tf.summary.scalar("disc_loss", disc_loss_tracker.result(), step=epoch)

        # Validation.
        epoch_valid_start = time()
        for step, batch in valid_ds.enumerate():
            if len(batch) == 3:
                input_image, target, weights = batch
            else:
                input_image, target = batch
                weights = None

            (
                val_gen_total_loss,
                val_gen_gan_loss,
                val_gen_l1_loss,
                val_disc_loss,
            ) = eval_step(
                generator,
                discriminator,
                input_image,
                target,
                weights=weights,
            )

            val_gen_total_loss_tracker.update_state(val_gen_total_loss)
            val_gen_gan_loss_tracker.update_state(val_gen_gan_loss)
            val_l1_loss_tracker.update_state(val_gen_l1_loss)
            val_disc_loss_tracker.update_state(val_disc_loss)

        valid_duration = time() - epoch_valid_start
        valid_duration = f"{valid_duration//60:.0f}m {valid_duration%60:.0f}s"
        print(
            valid_template.format(
                valid_duration,
                val_gen_total_loss_tracker.result(),
                val_gen_gan_loss_tracker.result(),
                val_l1_loss_tracker.result(),
                val_disc_loss_tracker.result(),
            )
        )

        with valid_writer.as_default():
            tf.summary.scalar(
                "gen_total_loss", val_gen_total_loss_tracker.result(), step=epoch
            )
            tf.summary.scalar(
                "gen_gan_loss", val_gen_gan_loss_tracker.result(), step=epoch
            )
            tf.summary.scalar("gen_l1_loss", val_l1_loss_tracker.result(), step=epoch)
            tf.summary.scalar("disc_loss", val_disc_loss_tracker.result(), step=epoch)

        if val_gen_total_loss_tracker.result() < best_val_gen_total_loss:
            print(
                f"New best val_gen_total_loss ({best_val_gen_total_loss:.5f} -> "
                f"{val_gen_total_loss_tracker.result():.5f}). Saving checkpoint at epoch {epoch+1}"
            )
            best_val_gen_total_loss = val_gen_total_loss_tracker.result()
            ckpt_manager.save()

        # Reset metrics in the end of every epoch.
        gen_total_loss_tracker.reset_state()
        gen_gan_loss_tracker.reset_state()
        gen_l1_loss_tracker.reset_state()
        disc_loss_tracker.reset_state()
        val_gen_total_loss_tracker.reset_state()
        val_gen_gan_loss_tracker.reset_state()
        val_l1_loss_tracker.reset_state()
        val_disc_loss_tracker.reset_state()


# Steps.
# ------------------------------------------------


@tf.function
def train_step(
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    input_image,
    target,
    weights=None,
):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target, weights=weights
        )
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss


@tf.function
def eval_step(
    generator,
    discriminator,
    input_image,
    target,
    weights=None,
):
    gen_output = generator(input_image, training=False)

    disc_real_output = discriminator([input_image, target], training=False)
    disc_generated_output = discriminator([input_image, gen_output], training=False)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
        disc_generated_output, gen_output, target, weights=weights
    )
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss


# Losses.
# ------------------------------------------------

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
l1 = tf.keras.losses.MeanAbsoluteError()


def generator_loss(disc_generated_output, gen_output, target, weights=None):
    LAMBDA = 100
    gan_loss = bce(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = l1(target, gen_output, sample_weight=weights)
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = bce(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = bce(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


# Utilities.
# ------------------------------------------------


def create_experiment_desc(config: ml_collections.ConfigDict) -> str:
    args = [os.path.basename(config.dset_dir)]
    if config.dataloader.class_weights:
        args.append(config.dataloader.weights_key)
    args.extend(
        [
            "pix2pix",
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
