"""pytest configuration — adds the backend package to sys.path so all test
modules can import from services, models, etc. without installing the package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
