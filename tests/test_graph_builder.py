import pytest
from src.data.step_parser import STEPParser
from src.data.graph_builder import BRepGraphBuilder

def test_build_graph():
    parser = STEPParser()
    shape = parser.parse("tests/data/sample.stp")
    builder = BRepGraphBuilder()
    graph = builder.build(shape)

    assert graph.num_nodes > 0
    assert graph.num_edges > 0
    assert graph.x.shape[1] == 16  # node features
    assert graph.edge_attr.shape[1] == 11  # edge features