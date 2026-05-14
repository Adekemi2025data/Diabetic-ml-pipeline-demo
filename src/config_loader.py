import yaml
import os

def load_config(path="config/config.yaml"):
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
    with open(path, "r") as f:
        return yaml.safe_load(f)
