"""Shim: re-exports stdlib logging so this file doesn't shadow it.

This file must stay named logging.py for backwards compatibility, but the
real implementation has moved to log_config.py.
"""
import importlib.util
import os
import sys

# Load real stdlib logging directly from its file path, bypassing sys.path
_stdlib_dir = os.path.dirname(os.__file__)
_log_init = os.path.join(_stdlib_dir, "logging", "__init__.py")
_spec = importlib.util.spec_from_file_location("logging", _log_init)
_real = importlib.util.module_from_spec(_spec)
# Register before exec so recursive imports inside logging itself resolve correctly
sys.modules["logging"] = _real
_spec.loader.exec_module(_real)  # type: ignore[union-attr]
globals().update(vars(_real))
