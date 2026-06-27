"""Smoke tests — simulate post-deploy health and prediction checks."""
import json
import pytest

# conftest.py provides the `client` fixture and handles path/cwd setup.

VALID_PAYLOAD = {
    "features": {
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
}


def test_smoke_health(client):
    """GET /health → 200 and body contains 'status' key."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data


def test_smoke_predict_responds(client):
    """POST valid payload to /predict_single → 200 and body contains 'prediction' key."""
    response = client.post(
        "/predict_single",
        data=json.dumps(VALID_PAYLOAD),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data


def test_smoke_model_info(client):
    """GET /model_info → 200 and body contains 'best_model' key."""
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.get_json()
    assert "best_model" in data
