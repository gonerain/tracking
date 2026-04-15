from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]


def bootstrap(entrypoint: Callable[[], None]) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    entrypoint()
