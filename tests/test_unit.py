"""Unit tests for preprocessing and model wrapper functions in frontend/app.py."""
import sys
import os
import numpy as np
import pytest

# conftest.py already sets cwd to frontend/ and inserts it into sys.path
from app import preprocess_features, predict_with_model

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

GOODWARE_DICT = {
    "BaseOfCode": 4096, "BaseOfData": 40960, "Characteristics": 783,
    "DllCharacteristics": 0, "Entropy": 5.981248597142612,
    "FileAlignment": 512, "ImageBase": 4194304, "Machine": 332,
    "Magic": 267, "NumberOfRvaAndSizes": 16, "NumberOfSections": 5,
    "NumberOfSymbols": 0, "PE_TYPE": 267, "PointerToSymbolTable": 0,
    "Size": 178688, "SizeOfCode": 33792, "SizeOfHeaders": 1024,
    "SizeOfImage": 33759232, "SizeOfInitializedData": 177664,
    "SizeOfOptionalHeader": 224, "SizeOfUninitializedData": 33557504,
    "TimeDateStamp": 0,
}

ALL_FEATURE_KEYS = list(GOODWARE_DICT.keys())


# ---------------------------------------------------------------------------
# test_preprocess_returns_array
# ---------------------------------------------------------------------------

def test_preprocess_returns_array():
    """preprocess_features with a complete 22-key dict should return shape (1, N) where N >= 22."""
    result = preprocess_features(GOODWARE_DICT)
    assert hasattr(result, "shape"), "Result should be a numpy array with a .shape attribute"
    assert result.shape[0] == 1, f"Expected 1 row, got {result.shape[0]}"
    assert result.shape[1] >= 22, f"Expected at least 22 columns, got {result.shape[1]}"


# ---------------------------------------------------------------------------
# test_preprocess_fills_nan
# ---------------------------------------------------------------------------

def test_preprocess_fills_nan():
    """preprocess_features should fill missing/None values with 0, producing no NaN."""
    features_with_none = dict(GOODWARE_DICT)
    features_with_none["Entropy"] = None  # introduce a None value

    result = preprocess_features(features_with_none)
    assert not np.isnan(result).any(), "Result should contain no NaN values after preprocessing"


# ---------------------------------------------------------------------------
# test_preprocess_coerces_string_to_numeric
# ---------------------------------------------------------------------------

def test_preprocess_coerces_string_to_numeric():
    """preprocess_features should coerce string numeric values without raising an error."""
    features_with_string = dict(GOODWARE_DICT)
    features_with_string["FileAlignment"] = "512"  # string instead of int

    # Should not raise; result should be a numeric array
    result = preprocess_features(features_with_string)
    assert result.dtype.kind in ("f", "i", "u"), (
        f"Expected numeric dtype, got {result.dtype}"
    )


# ---------------------------------------------------------------------------
# test_predict_returns_valid_label
# ---------------------------------------------------------------------------

def test_predict_returns_valid_label():
    """predict_with_model should return exactly 'malware' or 'goodware'."""
    features_scaled = preprocess_features(GOODWARE_DICT)
    label = predict_with_model(features_scaled)
    assert label in ("malware", "goodware"), (
        f"Expected 'malware' or 'goodware', got {label!r}"
    )


# ---------------------------------------------------------------------------
# test_preprocess_all_zeros
# ---------------------------------------------------------------------------

def test_preprocess_all_zeros():
    """preprocess_features with all-zero values should return a numpy array with no NaN."""
    zero_dict = {key: 0 for key in ALL_FEATURE_KEYS}
    result = preprocess_features(zero_dict)
    assert isinstance(result, np.ndarray), "Result should be a numpy ndarray"
    assert not np.isnan(result).any(), "All-zero input should not produce any NaN"
