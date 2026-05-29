import json
import os
from typing import Any, Optional


def load_data(
    path: str,
    n_samples: Optional[int] = None,
    key: Optional[str] = None,
) -> list[Any]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]
    if key is not None:
        data = [item[key] for item in data]
    if n_samples is not None:
        data = data[:n_samples]
    return data


def count_existing_samples(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)
