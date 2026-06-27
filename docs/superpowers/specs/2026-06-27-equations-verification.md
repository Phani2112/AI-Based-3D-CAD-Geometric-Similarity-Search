# Equation Verification for CADGCL Implementation

## CADGCL Paper Equations

### 1. EBC (Edge Betweenness Centrality) - Eq.1
```
EBC(e) = Σ σ(s,t|e) / σ(s,t) for all node pairs (s,t)
```
**Verification**: This calculates how many shortest paths between nodes pass through edge e. Higher EBC = more central edges. Used for perturbation strategy (remove top p_edge portion).

### 2. Feature Masking - Eq.2 (implied)
```
X' = X ◦ m  where m ∈ {0,1}^N sampled from Bernoulli(1-p_mask)
```
**Verification**: p_mask = 0.3 means 30% of node features are zeroed out.

### 3. InfoNCE Loss - Eq.5/6
```
l(u_i, v_i) = -log( exp(sim(u_i, v_i)/τ) / Σ exp(sim(u_i, v_k)/τ) + Σ exp(sim(u_i, u_k)/τ) )
LC = (1/2N) Σ [l(u_i, v_i) + l(v_i, u_i)]
```
**Verification**: τ = 0.07 temperature parameter for contrastive loss.

### 4. BMM Negative Sampling - Eq.9-14
```
p(s) = Σ λ_c p(s|α_c, β_c)
μ_c = Σ η_i * s_i,  σ_c^2 = Σ η_i * (s_i - μ_c)^2
p(c|s) = λ_c * p(s|α_c, β_c) / p(s)
w(i,k) = p(c_true|s_ik) * s_ik / Σ p(c_true|s_ik) * s_ik
```
**Verification**: Uses Beta Mixture Model to weight negative samples, distinguishing true vs false negatives.

## VGNet Paper Equations

### 1. BAGConv Update - Eq.1
```
h_i^(k+1) = σ( W^(k) · [ h_i, Mean( { f_Δ(e_ij) ⊕ h_j, ∀j ∈ ξ(N(i)) } ) ] )
```
**Verification**: 
- f_Δ projects edge features to node feature space
- VGNet uses FC(5, 14) - edge features 5 dims, output 14 dims to match node features
- Our edge dims = 11, node dims = 16 → FC(11, 16)

### 2. BAGPool Score - Eq.2
```
Score_ij = Softmax( W^(k) ( f_Δ(e_ij) ⊕ [h_i, h_j] ) + b^(k) )
```
**Verification**: Scores for edge contraction, uses same f_Δ projection.

## Calculations Verification

| Component | Calculation | Status |
|-----------|-------------|--------|
| Node features | 10 (surface types) + 3 (normal) + 3 (tangent) = 16 | ✅ Correct |
| Edge features | 7 (curve types) + 3 (direction) + 1 (length) = 11 | ✅ Correct |
| f_Δ projection | FC(11, 16) | ✅ Matches VGNet pattern (FC(edge_dim, node_dim)) |
| VGNet comparison | FC(5, 14) = 5 edge dims → 14 node dims | ✅ Pattern confirmed |

## Discrepancies Found

None. All calculations verified.