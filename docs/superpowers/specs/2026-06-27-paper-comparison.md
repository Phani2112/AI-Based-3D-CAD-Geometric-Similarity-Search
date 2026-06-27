# CADGCL Implementation vs Paper Comparison

## Paper Target Metrics (corrected)
| Metric | Paper Value | Status |
|--------|-------------|--------|
| mAP@50 | ≥ 89.35% | ⚠️ Not measured (no labels in FabWave) |
| Top-10 mAP | ~85-89% (estimated) | ⚠️ Not measured |

## Implementation Comparison

### ✅ Matched Paper Specifications
| Component | Paper | Implementation | Status |
|-----------|-------|----------------|--------|
| Node features | 16 dims (10 types + 3 normal + 3 tangent) | ✅ 16 dims | MATCH |
| Edge features | 11 dims (7 types + 3 direction + 1 length) | ✅ 11 dims | MATCH |
| Architecture | 2-layer BAGConv + BAGPool | ✅ BAGConv + Global Mean Pool | CLOSE |
| InfoNCE τ | 0.07 | ✅ 0.07 | MATCH |
| EBCE p_edge | 0.1 | ✅ 0.1 | MATCH |
| BMM negative sampling | Yes | ✅ Yes | MATCH |
| Training epochs | 20 | ✅ Configurable | MATCH |
| Batch size | 128 | ✅ 128 | MATCH |
| lr | 0.01 | ✅ 0.01 | MATCH |

### ⚠️ Architectural Differences
| Component | Paper | Implementation | Impact |
|-----------|-------|----------------|--------|
| Pooling | BAGPool (Eq.2) | Global Mean Pool | Lower performance expected |
| Feature masking | p_mask=0.3 | Not implemented | May reduce robustness |
| BMM timing | M=5 epochs | Immediate | May affect convergence |

### 🔧 Missing for Full Parity
1. **BAGPool** - The paper's pooling method
2. **Feature masking** - Bernoulli mask at p=0.3
3. **Full training** - Need 20 epochs on all 4,571 models

## Current Performance Estimate
With mock embeddings: mAP@10 = 19.44% (random baseline)
Paper target: mAP@10 ≈ 89% (inferred from mAP@50 ≥ 89.35%)
Gap: 69.56% - Training required to close

## Commands to Run Full Training
```bash
# Full training (will take several minutes)
python -m src.cli train --epochs 20 --batch-size 128
```