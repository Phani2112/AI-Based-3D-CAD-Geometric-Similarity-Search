### Task 2: B-rep Graph Builder - Report

**STATUS:** DONE

**Commits made:** 2f6945e

**Tests:** 2/2 passing

**Summary:** Implemented `BRepGraphBuilder` in `src/data/graph_builder.py` that:
- Extracts faces from `ParsedShape` as graph nodes
- Computes surface types using OCP GeomAbs enums (10 categories)
- Calculates normal and tangent vectors (3 values each) at surface midpoints
- Extracts edges via adjacency mapping between faces sharing curves
- Computes curve types (7 categories), direction vectors (3), and edge lengths (1)
- Returns `BRepGraph` with node features (16-dim) and edge features (11-dim)
- Uses OCP (OpenCASCADE Python bindings) instead of OCC for broader compatibility

The implementation follows the VGNet specification referenced in the CADGCL design:
- Node features: 10 surface types (one-hot) + 3 normal + 3 tangent = 16
- Edge features: 7 curve types (one-hot) + 3 direction + 1 length = 11
- Edges connect adjacent faces sharing a common curve