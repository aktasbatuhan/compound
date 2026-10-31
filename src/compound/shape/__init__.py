"""Workload shape: what traffic looks like (sizes, timing, prompt reuse) without its text."""

from compound.shape.loaders import LoadReport, load_shapes
from compound.shape.profile import CacheAssumptions, build_profile, format_profile
from compound.shape.records import CallShape, new_salt, shape_from_messages

__all__ = [
    "CacheAssumptions",
    "CallShape",
    "LoadReport",
    "build_profile",
    "format_profile",
    "load_shapes",
    "new_salt",
    "shape_from_messages",
]
