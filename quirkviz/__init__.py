"""Event displays for ATLAS quirk simulation: detector, sim hits, quirk trajectories."""

from .geometry import InnerDetector, ID_RUN3
from .ntuple import HitNtuple, Event, HitSet, TruthParticle
from .trajectory import (quirk_trajectories, trace_through_volume,
                         oscillation_scales, parse_g4_debug_log)
from .display import plot_event, plot_rphi, plot_rz, quirk_extent

__all__ = [
    "InnerDetector", "ID_RUN3",
    "HitNtuple", "Event", "HitSet", "TruthParticle",
    "quirk_trajectories", "trace_through_volume", "oscillation_scales",
    "parse_g4_debug_log",
    "plot_event", "plot_rphi", "plot_rz", "quirk_extent",
]
