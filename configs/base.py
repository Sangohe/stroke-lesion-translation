import ml_collections


def get_common_config():
    """Returns config values other than model parameters."""

    config = ml_collections.ConfigDict()
    config.gpu_num = "3"
    config.log_level = "3"
    config.root_dir = "/home/sangohe/projects/lesion-aware-translation/results/"

    config.random_seed = 2431
    config.use_ema = True
    config.slice_size = 224
    config.mixed_precision = False

    # Optimization.
    config.epochs = 600
    config.optimizer_name = "adamw"
    config.use_scheduler = True
    config.base_lr = 5e-4
    config.weight_decay = 1e-5
    config.grad_norm_clip = 1.0
    config.warmup_lr = 0.0
    config.warmup_epoch_percentage = 0.15

    # Will be set from ./models.py and ./dataloaders.py
    config.model = None
    config.dataloader = None

    return config
