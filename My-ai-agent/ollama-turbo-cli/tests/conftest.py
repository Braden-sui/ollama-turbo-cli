# Ensure the project root (containing the 'src' package) is importable during tests
import os
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import centralized logging/warning silencers for tests (side effects only)
try:
    from src.core import config as _core_config  # noqa: F401
except Exception:
    pass
