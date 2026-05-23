import os
from pathlib import Path

APP_ROOT_DIR: Path = Path(os.path.abspath(__file__)).parents[2]

DATA_DIR: Path = APP_ROOT_DIR / "data"

OUTPUT_DIR: Path = APP_ROOT_DIR / "outputs"
