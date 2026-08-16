"""RPG procedural validation-data generator for NoRefEval.

Fully deterministic, code-only generation of small validation sets for the
three evaluation modes (no-reference, endpoint-2x, full-reference).  See
PICPLAN.md for the complete specification.

Nothing in this package touches the network, external art assets, or any
generative model: every pixel is drawn by OpenCV from analytic motion.
"""

__version__ = "0.1.0"

GENERATOR_NAME = "rpg_validation_generator"
SCHEMA_VERSION = "rpg-procedural-validation-v1"
