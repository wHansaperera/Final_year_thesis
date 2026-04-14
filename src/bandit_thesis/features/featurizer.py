from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class SimpleFeaturizer:
    """
    Context-arm featurizer for persistent-user personalization.

    Features:
    - one-hot user segment
    - one-hot user id
    - one-hot arm
    - transient numeric context (device, weekend, hour bucket)

    The FM learns user-arm interactions from the user-id and arm one-hots.
    The linear probit baseline can still generalize through segment and arm
    main effects during cold start.
    """

    n_segments: int
    n_users: int
    n_arms: int

    @property
    def dim(self) -> int:
        return self.n_segments + self.n_users + self.n_arms + 3

    def transform(self, context: Dict[str, Any], arm: int) -> np.ndarray:
        z = np.zeros(self.dim, dtype=float)

        seg = int(context["segment_id"])
        user_id = int(context["user_id"])
        arm = int(arm)

        seg_base = 0
        user_base = seg_base + self.n_segments
        arm_base = user_base + self.n_users
        numeric_base = arm_base + self.n_arms

        z[seg_base + seg] = 1.0
        z[user_base + user_id] = 1.0
        z[arm_base + arm] = 1.0
        z[numeric_base + 0] = float(context["device"])
        z[numeric_base + 1] = float(context["is_weekend"])
        z[numeric_base + 2] = float(context["hour_bucket"]) / 5.0
        return z
