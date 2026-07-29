"""Build-time metadata.

Wheel/sdist builds replace ``BUILD_COMMIT`` in the build output. Source-tree
runs keep ``unknown`` and use the live Git checkout instead.
"""

BUILD_COMMIT = "unknown"
BUILD_DIRTY = None
