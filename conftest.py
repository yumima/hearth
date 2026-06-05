import sys
from pathlib import Path

# Make `hearth` importable from src/ even without an editable install, and the
# project root importable so `from tests.test_routes import ...` resolves.
ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
