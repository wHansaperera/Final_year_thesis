from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RNGStreams:
    rng_env: np.random.Generator
    rng_agent: np.random.Generator
    rng_baseline: np.random.Generator


def make_rng_streams(seed: int) -> RNGStreams:
    """
    IMPORTANT:
    - env RNG must NOT be shared with agents
    - ensures fairness across models
    """
    return RNGStreams(
        rng_env=np.random.default_rng(seed + 1000),
        rng_agent=np.random.default_rng(seed + 2000),
        rng_baseline=np.random.default_rng(seed + 3000),
    )
