# Ensure the project root (containing the 'src' package) is importable during tests
import os
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
