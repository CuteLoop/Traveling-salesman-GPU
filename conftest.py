# conftest.py — adds the project root to sys.path so that
# `from baselines.ga_runner import ...` works in tests without installing the package.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
