### Task 2: B-rep Graph Builder

**Files:**
- Create: `src/data/graph_builder.py`
- Test: `tests/test_graph_builder.py`

**Interfaces:**
- Consumes: `TopoDS_Shape` from STEPParser
- Produces: `BRepGraph` with nodes (features) and edges (adjacency)

**Step 1: Write the failing test**

```python
def test_build_graph():
    from src.data.step_parser import STEPParser
    from src.data.graph_builder import BRepGraphBuilder
    
    parser = STEPParser()
    shape = parser.parse("tests/data/sample.stp")
    builder = BRepGraphBuilder()
    graph = builder.build(shape)
    
    assert graph.num_nodes > 0
    assert graph.num_edges > 0
    assert graph.x.shape[1] == 16  # node features
    assert graph.edge_attr.shape[1] == 11  # edge features
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph_builder.py -v`
Expected: FAIL

**Step 3: Write minimal implementation (surface types)**

```python
# src/data/graph_builder.py
SURFACE_TYPES = {
    'PLANE': 0, 'CYLINDRICAL_SURFACE': 1, 'CONICAL_SURFACE': 2,
    'TOROIDAL_SURFACE': 3, 'SPHERICAL_SURFACE': 4,
    'B_SPLINE_SURFACE_WITH_KNOTS': 5, 'RATIONAL_B_SPLINE_SURFACE': 6,
    'SURFACE_OF_REVOLUTION': 7, 'SURFACE_OF_EXTRUSION': 8,
    'OFFSET_SURFACE': 9
}

CURVE_TYPES = {
    'EDGE_CURVE': 0, 'B_SPLINE_CURVE_WITH_KNOTS': 1,
    'POLYLINE': 2, 'BEZIER_CURVE': 3, 'HYPERBOLA': 4,
    'PARABOLA': 5, 'ELLIPSE': 6
}

class BRepGraphBuilder:
    def build(self, shape):
        # Extract faces as nodes
        # Extract edges as adjacency
        # Calculate one-hot types + normal/tangent + direction/length
        pass
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_graph_builder.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/graph_builder.py tests/test_graph_builder.py
git commit -m "feat: add B-rep graph builder with fixed surface/curve types"
```

Report file: `/home/jose-draeger/workspace/CADGCL V2/.superpowers/sdd/task-2-report.md`