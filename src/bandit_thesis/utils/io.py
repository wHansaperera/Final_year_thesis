from __future__ import annotations
from typing import Any, Dict, Iterable, List
import json
import os
import pandas as pd
import numpy as np

def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def write_csv(path: str, df: pd.DataFrame) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)
