"""Record the exact software versions and hardware used (results/environment.json).
Usage: python src/env_info.py"""
import platform

import sklearn
import torch
import torchvision

from common import RESULTS_DIR, save_json, get_device

info = {"python": platform.python_version(), "os": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "scikit-learn": sklearn.__version__, "device": str(get_device()),
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
save_json(info, RESULTS_DIR / "environment.json")
for k, v in info.items():
    print(f"{k:13s} {v}")
