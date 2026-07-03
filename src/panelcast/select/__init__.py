"""Model-selection protocol (`panelcast select`).

The selection discipline performed manually on AOTY — candidate enumeration,
staged sweeps, paired held-out scoring, pre-registered promotion rules —
packaged as a portable feature. See issue #78 and the 0.7.0 milestone.
"""

from panelcast.select.prior_screen import (
    TransformScreen,
    render_prior_block,
    screen_transforms,
)
from panelcast.select.space import (
    EXCLUDED_FIELDS,
    KNOBS,
    Knob,
    arm_conflicts,
    default_arm,
    enumerate_space,
    knob_is_active,
)

__all__ = [
    "EXCLUDED_FIELDS",
    "KNOBS",
    "Knob",
    "TransformScreen",
    "arm_conflicts",
    "default_arm",
    "enumerate_space",
    "knob_is_active",
    "render_prior_block",
    "screen_transforms",
]
