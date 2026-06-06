"""CLI: evaluacion downstream del modelo. Wrapper de actas_cnn.evaluate.

Uso:
  python scripts/evaluate.py --split val
  python scripts/evaluate.py --split test --checkpoint checkpoints/resnet18_best.pt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from actas_cnn.evaluate import main

if __name__ == "__main__":
    main()
