"""smartevsim

This repository contains the code to simulate coordination of EV clusters for
the purpose of testing control algorithms.

The package exposes a small, stable public API here so callers can do:

	from smartevsim import Engine, EV, SimConfig

Only lightweight imports are done at package import time to keep startup fast.
"""

__version__ = "0.1.0"

# Re-export the most commonly used classes from submodules as the public API.
from .engine import Engine
from .units.ev import EV
from .utils.data_classes import SimConfig, StepRecord, WorldState

__all__ = [
    "Engine",
    "EV",
    "SimConfig",
    "StepRecord",
    "WorldState",
    "__version__",
]
