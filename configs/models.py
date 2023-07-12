import ml_collections


def get_pix2pix_config():
    config = ml_collections.ConfigDict()
    config.name = "pix2pix"
    config.out_channels = 1
    config.out_activation = "sigmoid"
    return config
