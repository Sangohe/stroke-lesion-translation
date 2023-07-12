import ml_collections

import os

from configs import base
from configs import models
from configs import dataloaders


def get_config(model_dset_dir: str) -> ml_collections.ConfigDict:
    model, dset_dir = model_dset_dir.split(",")
    config = base.get_common_config()
    config.model_name = model
    config.dset_dir = dset_dir

    # Load the dataloader and model configs.
    config.dataloader = dataloaders.get_dataloader_config(os.path.abspath(dset_dir))
    get_model_config = getattr(models, f"get_{model}_config")
    config.model = get_model_config()

    return config
