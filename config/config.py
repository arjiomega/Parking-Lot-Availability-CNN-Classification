from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = Path(BASE_DIR, "data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR = Path(DATA_DIR, "training")