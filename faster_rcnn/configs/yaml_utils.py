import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r+", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data
