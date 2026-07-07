"""Backend auto-detection.

Import ``Backend`` from here to get the correct implementation
for the current platform.
"""

from auto_chasm.backends.base import Backend

__all__ = ["Backend"]
