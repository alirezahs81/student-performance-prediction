import pytest
from pathlib import Path
from src.data import DEFAULT_DATA_PATH

@pytest.fixture
def data_path():
    return DEFAULT_DATA_PATH
