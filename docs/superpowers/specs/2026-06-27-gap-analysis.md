# CADGCL Implementation Gap Analysis - VGNet Cross-Reference

| Phase/Step | Gap Description | VGNet Solution | CADGCL Relevance |
|------------|-----------------|----------------|------------------|
| **Phase 1.1** | What Python library for STEP parsing? | Uses STEP files as intermediate exchange format for geometric data | Confirms STEP files are input, but no specific library mentioned |
| **Phase 1.2** | What specific node features from each face? | **Surface type, normal vector, tangent vector** - VGNet uses these for face attributes and curve type/direction/length for edge attributes | ✅ DIRECTLY APPLICABLE - Use surface type, normal vector, tangent vector |
| **Phase 1.2** | How to determine adjacency (edges) between faces? | Edges represent "curve between intersecting surfaces" and "relationship between adjacent nodes" | ✅ DIRECTLY APPLICABLE - Edges connect adjacent faces |
| **Phase 2.1** | Activation and normalization for GCN layers? | Uses ReLU activation, **removes batch normalization** for structured CAD models, adds at output | ✅ DIRECTLY APPLICABLE - Remove batch norm from residual mapping, use ReLU |
| **Phase 2.1** | Layer normalization? Dropout? | Uses residual blocks with batch norm only at output, ReLU throughout | ✅ SPECIFIED - Use ReLU, batch norm only at block output |
| **Phase 2.2** | EBC algorithm for B-rep graphs? | No direct EBC method, but provides **BAGConv/BAGPool** for graph conv/pool | ⚠️ PARTIALLY RELEVANT - Use BAGConv instead of EBC for edge perturbation |
| **Phase 2.3** | BMM initialization method? | Uses **correlation loss function** for multimodal fusion, not BMM | ❌ NOT RELEVANT - BMM is CADGCL-specific, need paper info |
| **Phase 2.3** | Temperature parameter τ for InfoNCE? | Not specified in VGNet | ❌ NOT RELEVANT - VGNet uses correlation loss, not InfoNCE |
| **Phase 1.2** | Node feature dimension d? | Uses surface type classification as categorical features | ⚠️ PARTIALLY RELEVANT - Need to determine final dimension |

## Key VGNet Insights for CADGCL Implementation:

### Node Features (from VGNet):
- **Surface type** (plane, cylinder, sphere, etc.) - categorical
- **Normal vector** - 3 floats (x, y, z)
- **Tangent vector** - 3 floats (for curve-based surfaces)

### Edge Features (from VGNet):
- **Curve type** (line, arc, curve) - categorical
- **Direction** - vector information
- **Length** - scalar

### GCN Architecture (from VGNet):
- **Activation**: ReLU everywhere
- **Batch Norm**: Removed from residual mapping, kept at output
- **Architecture**: BAGConv (inspired by SAGEConv) + BAGPool (inspired by EdgePool)

## Remaining Gaps (Need User Input):

| Gap | Question for User |
|-----|-------------------|
| GCN layer type for CADGCL | Paper says GCN, VGNet uses BAGConv - which should we use? |
| Temperature τ value | Not in VGNet, what value should we use? (Paper says 0.07-0.1 typical) |
| Node feature dimension | Normal (3) + surface type (categories) + tangent (3) = d? |
| EBC algorithm details | VGNet doesn't specify betweenness centrality computation |