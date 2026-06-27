import pytest
import sys
import os

# Change working directory to frontend/ so that app.py can resolve model paths
# (app.py loads models relative to its own location via ../models/)
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'frontend'))

from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client
