from typing import *
import dataclasses

import numpy as np


@dataclasses.dataclass()
class BoundingBox:
    min: np.ndarray
    max: np.ndarray


@dataclasses.dataclass()
class SampleRegion:
    origin: np.ndarray
    rotation: np.ndarray
    gridding: np.ndarray
    spacing: np.ndarray
    mask: Optional[np.ndarray]
