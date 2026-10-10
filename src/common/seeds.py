"""Seed every source of randomness from config (AGENTS.md section 6).

`project.seed` drives Python's `random` and NumPy; scikit-learn and XGBoost take
`random_state` from the model blocks in config at construction time.
"""

from __future__ import annotations

import random
from typing import Any, Mapping

import numpy as np


def get_seed(cfg: Mapping[str, Any]) -> int:
    """Return `project.seed` from the config."""
    return int(cfg["project"]["seed"])


def seed_everything(cfg: Mapping[str, Any]) -> int:
    """Seed `random` and NumPy from `project.seed`.

    Args:
        cfg: Loaded config.

    Returns:
        The seed applied, for logging.
    """
    seed = get_seed(cfg)
    random.seed(seed)
    np.random.seed(seed)
    return seed
