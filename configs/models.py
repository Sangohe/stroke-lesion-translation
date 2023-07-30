import ml_collections


def get_pix2pix_config():
    config = ml_collections.ConfigDict()
    config.name = "pix2pix"
    config.out_channels = 1
    config.out_activation = "sigmoid"
    return config


def get_cyclegan_config():
    config = ml_collections.ConfigDict()
    config.name = "cyclegan"
    config.input_shape = (256, 256, 1)

    # Generators.
    config.use_casnet_generator = True
    config.out_channels = 1
    config.out_activation = "sigmoid"
    # casnet generator.
    config.n_blocks = 2
    config.use_instance_norm = True
    # vanilla generator.
    config.use_resize_convolution = False
    config.use_dropout = False

    # Discriminators.
    config.use_bias = True
    config.use_patchgan = True
    return config
