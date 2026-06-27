"""Integration tests for the Flask API endpoints."""
import json
import pytest

# conftest.py provides the `client` fixture and handles path/cwd setup.

# ---------------------------------------------------------------------------
# Sample payloads
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

MALWARE_DICT = {
    "BaseOfCode": 4096, "BaseOfData": 40960, "Characteristics": 783,
    "DllCharacteristics": 32768, "Entropy": 7.999899841773473,
    "FileAlignment": 512, "ImageBase": 4194304, "Machine": 332,
    "Magic": 267, "NumberOfRvaAndSizes": 16, "NumberOfSections": 5,
    "NumberOfSymbols": 0, "PE_TYPE": 267, "PointerToSymbolTable": 0,
    "Size": 11415344, "SizeOfCode": 37888, "SizeOfHeaders": 1024,
    "SizeOfImage": 192512, "SizeOfInitializedData": 124416,
    "SizeOfOptionalHeader": 224, "SizeOfUninitializedData": 0,
    "TimeDateStamp": 708992537,
}


# ---------------------------------------------------------------------------
# /predict_single
# ---------------------------------------------------------------------------

def test_predict_single_goodware(client):
    """POST goodware payload → 200 with valid success/prediction keys."""
    response = client.post(
        "/predict_single",
        data=json.dumps({"features": GOODWARE_DICT}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["prediction"] in ("malware", "goodware")


def test_predict_single_malware(client):
    """POST malware payload → 200 with success=True."""
    response = client.post(
        "/predict_single",
        data=json.dumps({"features": MALWARE_DICT}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True


def test_predict_single_missing_features(client):
    """POST empty features dict → 200 and 'prediction' key present (app fills missing with 0)."""
    response = client.post(
        "/predict_single",
        data=json.dumps({"features": {}}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    """GET /health → 200 with status == 'healthy'."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"


# ---------------------------------------------------------------------------
# / (index page)
# ---------------------------------------------------------------------------

def test_index_page(client):
    """GET / → 200 and HTML contains 'Malware Detection'."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Malware Detection" in response.data
