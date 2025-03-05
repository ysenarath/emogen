import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

__all__ = [
    "config",
]

output_dir = Path(__file__).parent.parent / "outputs"
output_dir.mkdir(exist_ok=True)

config = {
    "train_dir": output_dir / "train",
    "optimize_dir": output_dir / "optimize",
}


def set_seed(seed: Optional[int] = None):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
