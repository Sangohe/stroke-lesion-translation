import ml_collections


def get_unet_config():
    config = ml_collections.ConfigDict()
    config.name = "unet"
    config.input_shape = (128, 128, 1)
    config.out_channels = 1
    config.dim = 128
    config.out_activation = "tanh"
    config.groups = 8
    config.init_dim = 64
    config.dim_mults = [1, 2, 4, 8]
    return config


def get_pix2pix_config():
    config = ml_collections.ConfigDict()
    config.name = "pix2pix"
    config.out_channels = 1
    config.out_activation = "tanh"
    return config


def get_cyclegan_config():
    config = ml_collections.ConfigDict()
    config.name = "cyclegan"
    config.input_shape = (128, 128, 1)

    # Generators.
    config.use_casnet_generator = True
    config.out_channels = 1
    config.out_activation = "tanh"
    # casnet generator.
    config.n_blocks = 1
    config.use_instance_norm = True
    # vanilla generator.
    config.use_resize_convolution = False
    config.use_dropout = False

    # Discriminators.
    config.use_bias = True
    config.use_patchgan = True
    return config
