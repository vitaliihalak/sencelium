"""Loads the released training script and configuration for one scale, and reads the common arguments.

Usage inside an analysis script:  SCALE, CKPT = _scale.args(); tsm = _scale.load(SCALE)
Command line:  python analysis/<script>.py 65M [--checkpoint path/to/best.ckpt] [script-specific arguments]
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def args():
    scale = sys.argv.pop(1)
    ckpt = f"checkpoints/sencelium_{scale}/best.ckpt"
    if "--checkpoint" in sys.argv:
        i = sys.argv.index("--checkpoint")
        ckpt = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    return scale, ckpt


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load(scale):
    module = _load(f"train_sencelium_{scale}", ROOT / "scripts" / "train_sencelium.py")
    cfg = module.load_config(["--config", str(ROOT / "configs" / f"{scale}.yaml")])
    module.Config = lambda: cfg
    return module


def load_transformer(scale):
    return _load(f"train_transformer_{scale}", ROOT / "scripts" / "transformer" / f"train_transformer_{scale}.py")
