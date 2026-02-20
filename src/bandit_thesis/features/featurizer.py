from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np


@dataclass
class SimpleFeaturizer:
    """
    Converts context + arm -> feature vector z.

    Design:
    - One-hot segment (S)
    - One-hot arm (A)
    - small numeric features: device, weekend, hour_bucket (scaled)
    """

    n_segments: int
    n_arms: int

    @property
    def dim(self) -> int:
        return self.n_segments + self.n_arms + 3

    def transform(self, context: Dict[str, Any], arm: int) -> np.ndarray:
        z = np.zeros(self.dim, dtype=float)

        seg = int(context["segment_id"])
        z[seg] = 1.0

        arm = int(arm)
        z[self.n_segments + arm] = 1.0

        # numeric tail
        base = self.n_segments + self.n_arms
        z[base + 0] = float(context["device"])
        z[base + 1] = float(context["is_weekend"])
        z[base + 2] = float(context["hour_bucket"]) / 5.0
        return z
