"""CLI: entrenamiento del modelo. Wrapper de actas_cnn.training.

Uso:
  python scripts/train.py --manifest data/manifest_train.csv --root data/crops_train \\
                          --arch resnet18 --epochs 20
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from actas_cnn.training import main

if __name__ == "__main__":
    main()
