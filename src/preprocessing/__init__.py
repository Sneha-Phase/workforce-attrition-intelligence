"""
src/preprocessing/__init__.py
==============================
Public API for the preprocessing package.
"""

from src.preprocessing.pipeline import (
    build_preprocessor,
    encode_target,
    get_feature_names,
    prepare_features,
    CONTINUOUS_FEATURES,
    ENGINEERED_BINARY,
    ENGINEERED_CONTINUOUS,
    ORDINAL_PASSTHROUGH,
    RAW_CONTINUOUS,
)
from src.preprocessing.splitter import split_data, SplitResult

__all__ = [
    "build_preprocessor",
    "encode_target",
    "get_feature_names",
    "prepare_features",
    "split_data",
    "SplitResult",
    "CONTINUOUS_FEATURES",
    "ENGINEERED_BINARY",
    "ENGINEERED_CONTINUOUS",
    "ORDINAL_PASSTHROUGH",
    "RAW_CONTINUOUS",
]
