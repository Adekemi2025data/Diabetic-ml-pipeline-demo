import yaml
import os

def load_config(path="config/config.yaml"):
    """Load YAML configuration file reliably in GitHub Actions and locally."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, path)

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
