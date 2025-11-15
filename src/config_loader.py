import yaml
import os
def load_config():
    base_dir = os.path.dirname(os.path.dirname(__file__))  # sube de src/utils → proyecto raíz
    path = os.path.join(base_dir, "configs", "config.yaml")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
