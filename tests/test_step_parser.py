import pytest
from src.data.step_parser import STEPParser

def test_parse_step_file():
    parser = STEPParser()
    shape = parser.parse("tests/data/sample.stp")
    assert shape is not None
    assert hasattr(shape, 'Faces')
    assert len(shape.Faces) > 0