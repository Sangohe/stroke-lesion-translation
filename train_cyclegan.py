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
from models.cyclegan import CasnetGenerator, Generator, Discriminator
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
    if config.model.use_casnet_generator:
        print("Building Casnet generators...")
        generator_g = CasnetGenerator(
            config.model.input_shape,
            config.model.out_channels,
            config.model.out_activation,
            config.model.n_blocks,
            config.model.use_instance_norm,
        )
        generator_f = CasnetGenerator(
            config.model.input_shape,
            config.model.out_channels,
            config.model.out_activation,
            config.model.n_blocks,
            config.model.use_instance_norm,
        )
    else:
        print("Building vanilla generators...")
        generator_g = Generator(
            config.model.input_shape,
            config.model.out_channels,
            config.model.out_activation,
        )
        generator_f = Generator(
            config.model.input_shape,
            config.model.out_channels,
            config.model.out_activation,
        )

    discriminator_x = Discriminator(
        config.model.input_shape,
        config.model.use_bias,
        config.model.use_patchgan,
    )
    discriminator_y = Discriminator(
        config.model.input_shape,
        config.model.use_bias,
        config.model.use_patchgan,
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

    generator_g_optimizer = tf.keras.optimizers.Adam(lrs, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(lrs, beta_1=0.5)
    discriminator_x_optimizer = tf.keras.optimizers.Adam(lrs, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(lrs, beta_1=0.5)

    fit(
        config.weights_dir,
        config.logs_dir,
        train_dset,
        valid_dset,
        config.epochs,
        generator_g,
        generator_f,
        discriminator_x,
        discriminator_y,
        generator_g_optimizer,
        generator_f_optimizer,
        discriminator_x_optimizer,
        discriminator_y_optimizer,
    )

    print("Ending training...")


def fit(
    weights_dir: str,
    logs_dir: str,
    train_ds: tf.data.Dataset,
    valid_ds: tf.data.Dataset,
    epochs: int,
    generator_g: tf.keras.Model,
    generator_f: tf.keras.Model,
    discriminator_x: tf.keras.Model,
    discriminator_y: tf.keras.Model,
    generator_g_optimizer: tf.keras.optimizers.Optimizer,
    generator_f_optimizer: tf.keras.optimizers.Optimizer,
    discriminator_x_optimizer: tf.keras.optimizers.Optimizer,
    discriminator_y_optimizer: tf.keras.optimizers.Optimizer,
):
    metric_names = [
        "total_gen_g_loss",
        "total_gen_f_loss",
        "gen_g_loss",
        "gen_f_loss",
        "cycle_x_loss",
        "cycle_y_loss",
        "total_cycle_loss",
        "identity_y",
        "identity_x",
        "disc_y_loss",
        "disc_x_loss",
        "paired_y_loss",
    ]

    # Configure checkpoint manager.
    ckpt = tf.train.Checkpoint(
        generator_g=generator_g,
        generator_f=generator_f,
        discriminator_x=discriminator_x,
        discriminator_y=discriminator_y,
        generator_g_optimizer=generator_g_optimizer,
        generator_f_optimizer=generator_f_optimizer,
        discriminator_x_optimizer=discriminator_x_optimizer,
        discriminator_y_optimizer=discriminator_y_optimizer,
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, weights_dir, max_to_keep=3)

    train_writer = tf.summary.create_file_writer(os.path.join(logs_dir, "train"))
    valid_writer = tf.summary.create_file_writer(os.path.join(logs_dir, "valid"))

    train_template = "Train [{}] -> " + ", ".join(
        [m + ": {:.4f}" for m in metric_names]
    )
    valid_template = "Validation [{}] -> " + ", ".join(
        ["val_" + m + ": {:.4f}" for m in metric_names]
    )

    # Define metrics.
    train_metrics = {name: tf.keras.metrics.Mean(name=name) for name in metric_names}
    valid_metrics = {
        name: tf.keras.metrics.Mean(name="val_" + name) for name in metric_names
    }
    best_comparison_metric = tf.constant(np.inf)

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")

        # Training.
        epoch_train_start = time()
        for step, batch in train_ds.enumerate():
            if len(batch) == 3:
                real_x, real_y, weights = batch
            else:
                real_x, real_y = batch
                weights = None

            train_step_metrics = train_step(
                generator_g,
                generator_f,
                discriminator_x,
                discriminator_y,
                generator_g_optimizer,
                generator_f_optimizer,
                discriminator_x_optimizer,
                discriminator_y_optimizer,
                real_x,
                real_y,
                weights=weights,
            )

            for metric_name in metric_names:
                train_metrics[metric_name].update_state(train_step_metrics[metric_name])

        train_duration = time() - epoch_train_start
        train_duration = f"{train_duration//60:.0f}m {train_duration%60:.0f}s"
        print(
            train_template.format(
                train_duration,
                *[train_metrics[metric_name].result() for metric_name in metric_names],
            )
        )

        with train_writer.as_default():
            tf.summary.scalar("lr", generator_g_optimizer.lr, step=epoch)
            for metric_name in metric_names:
                tf.summary.scalar(
                    metric_name, train_metrics[metric_name].result(), step=epoch
                )

        # Validation.
        epoch_valid_start = time()
        for step, batch in valid_ds.enumerate():
            if len(batch) == 3:
                real_x, real_y, weights = batch
            else:
                real_x, real_y = batch
                weights = None

            valid_step_metrics = eval_step(
                generator_g,
                generator_f,
                discriminator_x,
                discriminator_y,
                real_x,
                real_y,
                weights=None,  # ?: Should use None to compare losses of different runs.
            )

            for metric_name in metric_names:
                valid_metrics[metric_name].update_state(valid_step_metrics[metric_name])

        valid_duration = time() - epoch_valid_start
        valid_duration = f"{valid_duration//60:.0f}m {valid_duration%60:.0f}s"
        print(
            valid_template.format(
                valid_duration,
                *[valid_metrics[metric_name].result() for metric_name in metric_names],
            )
        )

        with valid_writer.as_default():
            for metric_name in metric_names:
                tf.summary.scalar(
                    metric_name, valid_metrics[metric_name].result(), step=epoch
                )

        # comparison_metric = (
        #     valid_metrics["cycle_y_loss"].result()
        #     + valid_metrics["identity_y"].result()
        # )
        comparison_metric = valid_metrics["paired_y_loss"].result()
        if comparison_metric <= best_comparison_metric:
            print(
                f"New best paired_y_loss ({best_comparison_metric:.4f} -> "
                f"{comparison_metric:.4f}). Saving checkpoint at epoch {epoch+1}"
            )
            best_comparison_metric = comparison_metric
            ckpt_manager.save()

        # Reset metrics in the end of every epoch.
        for metric_name in metric_names:
            train_metrics[metric_name].reset_states()
            valid_metrics[metric_name].reset_states()


# Steps.
# ------------------------------------------------


@tf.function(jit_compile=True)
def train_step(
    generator_g,
    generator_f,
    discriminator_x,
    discriminator_y,
    generator_g_optimizer,
    generator_f_optimizer,
    discriminator_x_optimizer,
    discriminator_y_optimizer,
    real_x,
    real_y,
    weights=None,
):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        cycle_x_loss = calc_cycle_loss(real_x, cycled_x, weights=weights)
        cycle_y_loss = calc_cycle_loss(real_y, cycled_y, weights=weights)
        total_cycle_loss = cycle_x_loss + cycle_y_loss

        identity_y = identity_loss(real_y, same_y, weights=weights)
        identity_x = identity_loss(real_x, same_x, weights=weights)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_y
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_x

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(
        total_gen_g_loss, generator_g.trainable_variables
    )
    generator_f_gradients = tape.gradient(
        total_gen_f_loss, generator_f.trainable_variables
    )

    discriminator_x_gradients = tape.gradient(
        disc_x_loss, discriminator_x.trainable_variables
    )
    discriminator_y_gradients = tape.gradient(
        disc_y_loss, discriminator_y.trainable_variables
    )

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(
        zip(generator_g_gradients, generator_g.trainable_variables)
    )

    generator_f_optimizer.apply_gradients(
        zip(generator_f_gradients, generator_f.trainable_variables)
    )

    discriminator_x_optimizer.apply_gradients(
        zip(discriminator_x_gradients, discriminator_x.trainable_variables)
    )

    discriminator_y_optimizer.apply_gradients(
        zip(discriminator_y_gradients, discriminator_y.trainable_variables)
    )

    return {
        "total_gen_g_loss": total_gen_g_loss,
        "total_gen_f_loss": total_gen_f_loss,
        "gen_g_loss": gen_g_loss,
        "gen_f_loss": gen_f_loss,
        "cycle_x_loss": cycle_x_loss,
        "cycle_y_loss": cycle_y_loss,
        "total_cycle_loss": total_cycle_loss,
        "identity_y": identity_y,
        "identity_x": identity_x,
        "disc_y_loss": disc_y_loss,
        "disc_x_loss": disc_x_loss,
        "paired_y_loss": l1(real_y, fake_y, sample_weight=weights),
    }


@tf.function(jit_compile=True)
def eval_step(
    generator_g,
    generator_f,
    discriminator_x,
    discriminator_y,
    real_x,
    real_y,
    weights=None,
):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

        fake_y = generator_g(real_x, training=False)
        cycled_x = generator_f(fake_y, training=False)

        fake_x = generator_f(real_y, training=False)
        cycled_y = generator_g(fake_x, training=False)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=False)
        same_y = generator_g(real_y, training=False)

        disc_real_x = discriminator_x(real_x, training=False)
        disc_real_y = discriminator_y(real_y, training=False)

        disc_fake_x = discriminator_x(fake_x, training=False)
        disc_fake_y = discriminator_y(fake_y, training=False)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        cycle_x_loss = calc_cycle_loss(real_x, cycled_x, weights=weights)
        cycle_y_loss = calc_cycle_loss(real_y, cycled_y, weights=weights)
        total_cycle_loss = cycle_x_loss + cycle_y_loss

        identity_y = identity_loss(real_y, same_y, weights=weights)
        identity_x = identity_loss(real_x, same_x, weights=weights)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_y
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_x

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    return {
        "total_gen_g_loss": total_gen_g_loss,
        "total_gen_f_loss": total_gen_f_loss,
        "gen_g_loss": gen_g_loss,
        "gen_f_loss": gen_f_loss,
        "cycle_x_loss": cycle_x_loss,
        "cycle_y_loss": cycle_y_loss,
        "total_cycle_loss": total_cycle_loss,
        "identity_y": identity_y,
        "identity_x": identity_x,
        "disc_y_loss": disc_y_loss,
        "disc_x_loss": disc_x_loss,
        "paired_y_loss": l1(real_y, fake_y, sample_weight=weights),
    }


# Losses.
# ------------------------------------------------

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
l1 = tf.keras.losses.MeanAbsoluteError()
LAMBDA = 10


def calc_cycle_loss(real_image, cycled_image, weights=None):
    loss1 = l1(real_image, cycled_image, sample_weight=weights)
    return LAMBDA * loss1


def identity_loss(real_image, same_image, weights=None):
    loss = l1(real_image, same_image, sample_weight=weights)
    return LAMBDA * 0.5 * loss


def generator_loss(generated):
    return bce(tf.ones_like(generated), generated)


def discriminator_loss(real, generated):
    real_loss = bce(tf.ones_like(real), real)
    generated_loss = bce(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5


# Utilities.
# ------------------------------------------------


def create_experiment_desc(config: ml_collections.ConfigDict) -> str:
    args = [os.path.basename(config.dset_dir)]
    if config.dataloader.class_weights:
        args.append(config.dataloader.weights_key)
    args.extend(
        [
            "casnet" if config.model.use_casnet_generator else "cyclegan",
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
