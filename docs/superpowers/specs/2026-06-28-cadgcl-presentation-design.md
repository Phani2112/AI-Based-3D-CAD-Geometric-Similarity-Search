# CADGCL Final Presentation Design

**Date:** 2026-06-28  
**Deck:** Internal 1-hour company-focused presentation for Gühring  
**Format target:** HTML slides  
**Design system:** `Guhring DESIGN.md`  
**Confidentiality:** Internal/confidential company material. The deck may use Gühring dataset context, Similia material, and internal similarity definitions, and should be marked confidential.

## Presentation Strategy

Use a **Decision Journey** structure. The presentation should explain how the project moved from a business problem to research selection, then to a CADGCL prototype, and finally toward industrial validation with Gühring data.

The professor will be present, but the primary audience is the company. The deck should therefore stay company-focused while still showing enough research and technical rigor to justify the decisions.

Technical depth should be **executive-technical balance**: explain B-rep graphs, contrastive learning, BMM, implementation gaps, and evaluation visually, with limited equations. The live demo will happen outside the slide deck, so the slides should prepare and frame the demo rather than include it.

## Visual Direction

Follow the Gühring design system:

- Industrial, authoritative, structured, grid-aligned layouts.
- Primary red `#CF2E2E` for key highlights and actions.
- Secondary black `#1A1A1A` for headers, dividers, and strong contrast areas.
- White surfaces with restrained spacing and minimal rounding.
- Sans-serif typography, bold utilitarian headlines, uppercase labels for sections.
- Avoid decorative colors beyond red, black, white, and neutral grays.
- Include `Confidential | Internal Gühring Project` on relevant slides.

## Slide Outline

### 1. Cover

Title: `AI-Based Geometric Similarity Search for 3D CAD Models`  
Subtitle: `From research selection to CADGCL prototype`  
Include confidential marker. Use a dark machining/CAD-style hero treatment with a Gühring red accent.

### 2. Agenda

Sections:

- Introduction
- Research
- Building CADGCL
- Scaling Complexity
- Conclusions and Next Steps

### 3. Why This Project Exists

Message: Gühring has 670,000+ CAD models, but engineers still struggle to find reusable parts.

Key barriers:

- Fragmented search across PLM, CAD, ERP, and memory.
- Inconsistent or missing metadata.
- Local templates and knowledge barriers.
- Similia results are not accurate enough for the target workflow.

### 4. The Business Cost Of Not Finding Parts

Show the search decision flow:

`find existing design -> adapt proven component`  
`fail to find -> design from scratch`

Business impact:

- Duplicate components.
- Longer development cycles.
- Wasted engineering effort.
- Higher risk of avoidable design errors.

### 5. What Similarity Means At Gühring

Use the company definition from `dataset/datasetguhring/_meta/Similarity.pdf`.

Core message: similarity is context-dependent, not absolute. For cutting tools, the priority is:

`Full Similarity -> Functional Similarity -> Contour Similarity`

Definitions:

- Full similarity: cutting geometry, contour, and shank type match; size variants are acceptable.
- Functional similarity: same cutting geometry with different shank or secondary features; often the most realistic and valuable target.
- Contour similarity: only the outline or silhouette matches; useful fallback when cutting geometry differs.
- Key rule: comparisons are size-invariant, so absolute diameter and length are ignored.

### 6. Discovery Process

Show how company interviews and documents clarified the requirement.

Main findings:

- Geometry should be the primary search signal.
- Metadata should be optional post-search filtering only.
- Topology and cutting geometry matter more than simple dimensions.
- Localized similarity may become important.
- Siemens NX/NX-Open compatibility is a future constraint.
- Public datasets were needed first because of confidentiality limits.

### 7. Project Goals

Main question:

`Can current AI research provide a viable solution for Gühring's geometry-driven CAD similarity search problem?`

Supporting goals:

- Identify and compare relevant AI approaches.
- Select a model worth reproducing.
- Build a functional prototype.
- Evaluate whether the prototype produces useful similarity results.
- Prepare a comparison direction against Similia.

### 8. Research Strategy

Message: the review was designed to answer whether current AI research contains a viable path for Gühring's geometry-driven CAD similarity problem.

Emphasis: the review was biased toward recent work from 2020 onward because the goal was a practical prototype direction based on current methods, not a historical survey.

### 9. Initial Queries

Use the real poster search terms from `Poster Guhring.pdf`:

- `AI CAD similarity search`
- `CAD geometric similarity search`
- `ML geometric similarity search`

Message: the search started from the business need rather than from one pre-selected architecture.

### 10. The 7 Starting Papers

Placeholder-ready slide for the seven foundational papers to be added later.

Suggested table columns:

- Paper
- Year
- Representation
- Relevance to Gühring

Message: these seven papers formed the seed set for deeper citation mapping.

### 11. Research Rabbit Expansion

Explain Research Rabbit as the AI citation-mapping tool used to expand from the seven starting papers through references, citations, and similar-paper suggestions.

Use the network graph visual from the poster if appropriate.

Message: this helped avoid narrow search-term bias and gave a wider view of the research field.

### 12. Final Bibliography

Do not group the papers thematically. Show the final number of reviewed papers as a large-number visual.

Message: the bibliography was intentionally broad and recent, with emphasis on papers from 2020 onward, showing that the team cast a wide net before selecting the prototype architecture.

### 13. Pattern In The Literature: From Voxels To B-rep

Timeline slide.

Message: generic 3D methods such as voxels, point clouds, and multi-view approaches appeared often, but the research trend for CAD increasingly moved toward B-rep-native methods.

### 14. Why B-rep Won

Compare representation families:

- Voxel: simple but loses precision and is memory-heavy.
- Point cloud: flexible but weak topology and surface semantics.
- Multi-view: useful for visual recognition but not CAD-native.
- Mesh: dependent on tessellation and loses construction meaning.
- B-rep: preserves faces, edges, topology, surface types, and curve relationships.

Company relevance: Gühring needs functional cutting-tool similarity, not only visual resemblance.

### 15. Seminal Reference: UV-Net

Present UV-Net as a key landmark that pushed the field toward B-rep-native learning.

Message: UV-Net showed that CAD face, edge, and topology information can be used directly by neural models.

### 16. Overall Problems In The Field

Main limitations:

- Datasets are fragmented.
- Benchmarks differ.
- Similarity is defined differently across papers.
- Architectures are hard to compare.
- Reproducibility is limited by missing implementation details.

Add two or three paper references after the final bibliography is available.

Message: there was no obvious plug-and-play winner for industry use.

### 17. How We Selected A Model

Exclusion criteria:

- Not recent enough.
- Not CAD/B-rep-native.
- Not suitable for similarity retrieval.
- Too dependent on labels.
- Not reproducible enough.
- Not feasible with available datasets.

Ranking criteria:

- Direct relevance to similarity search.
- Unsupervised capability.
- B-rep support.
- Reproducibility.
- Dataset feasibility.
- Fit with Gühring's geometry-first requirements.

### 18. Top Candidates: BRepMAE And CADGCL

Side-by-side comparison.

- BRepMAE: strong B-rep representation learning and promising general pretraining direction.
- CADGCL: B-rep graph contrastive learning directly designed for unsupervised CAD retrieval.

Message: both were strong candidates, but CADGCL was more directly aligned with the search problem.

### 19. Selected Model: CADGCL

Message: CADGCL was chosen because it directly targets unsupervised CAD model retrieval from B-rep representations.

Important caveat: it was not chosen because it was easiest; it was chosen because it best matched Gühring's geometry-driven similarity search objective.

### 20. Before The Architecture: Two Important Disclaimers

Disclaimer 1: the project entered state-of-the-art research territory, beyond what could be implemented reliably from classroom knowledge alone.

Disclaimer 2: the CADGCL paper does not provide every technical detail required for full reproduction.

Message: the prototype is not just copying a paper; it required interpretation, engineering decisions, and cross-referencing related research.

### 21. How We Reconstructed The Missing Details

Investigation path:

`CADGCL paper -> missing implementation details -> author lineage / related work -> VGNet assumptions`

Message: when CADGCL did not define enough details, VGNet was used as the closest technical reference from the same research lineage.

### 22. CADGCL In One Slide

Show the three main parts:

- Data transformation and augmentation.
- Graph neural network encoder with contrastive learning.
- Beta Mixture Model negative sampling.

Message: CADGCL learns a vector representation where geometrically similar CAD models should end up close together.

### 23. Step 1: From B-rep To Graph

Show STEP/B-rep model on the left and graph representation on the right.

Graph definition:

- Faces/surfaces become nodes.
- Curves/adjacencies become edges.
- Node features include surface type, normal vector, tangent vector.
- Edge features include curve type, direction, and length.

Message: the model stays close to CAD geometry instead of reducing the part to pixels or points.

### 24. What The Graph Contains

Use a compact tensor table:

- Node matrix `X`: one row per face, 16 features.
- Edge index `edge_index`: connectivity between faces.
- Edge attributes `edge_attr`: one row per curve/edge, 11 features.
- Adjacency rule: connect faces that share a curve.

Message: the graph is a structured engineering representation, not a generic shape approximation.

### 25. Step 1.2: Two Graph Views

Explain the training pair:

- Original graph stays intact.
- Augmented graph is created with feature masking and edge betweenness centrality perturbation.

Message: the model learns stable similarity features by comparing the original CAD graph with an altered version of itself.

### 26. Step 2: Passing Original And Augmented Graphs Through The GNN

Show:

`G_original -> shared GNN -> z_original`  
`G_augmented -> same shared GNN -> z_augmented`

These two vectors form the positive pair because they come from the same CAD model.

### 27. How Contrastive Learning Works

Explain visually:

- Pull `z_original` and `z_augmented` together.
- Push embeddings from different CAD models apart.
- Training signal comes from augmented graph pairs, not labels.

### 28. The False Negative Problem

Message: in CAD datasets, two different files may actually be similar. Standard contrastive learning treats all other models as negatives, so it can wrongly push similar parts away from each other.

This is especially important for Gühring because the whole goal is to find reusable near-neighbors.

### 29. Step 3: BMM Negative Sampling

Explain Beta Mixture Model as the correction mechanism:

- It estimates which negative samples may actually be false negatives.
- Likely false negatives receive lower push-away pressure.
- Clear negatives are still pushed away.

Message: BMM makes the training objective more compatible with similarity search.

### 30. After Training: Embeddings Become Search Infrastructure

Separate the two embedding moments:

- Training phase: graph pairs produce embeddings, embeddings produce contrastive loss, loss updates GNN weights.
- Deployment/search phase: trained GNN generates one embedding per CAD model, embeddings are saved, a similarity index is built or loaded, and ranked retrieval becomes possible.

Message: training embeddings adjust the model; post-training embeddings become the searchable database.

### 31. Training Configuration

Core hyperparameters:

- Epochs: 20.
- Batch size: 128.
- Optimizer: Adam.
- Learning rate: 0.01.
- Temperature: 0.07.
- Feature mask rate: 0.3.
- Edge perturbation rate: 0.1.
- Embedding dimension: 256.

Message: these settings align as closely as possible with CADGCL and related research.

### 32. What The Paper Did Not Specify

Table columns:

- Missing detail.
- Why it mattered.
- How we resolved it.

Key rows:

- B-rep-to-graph construction logic -> required to create model input -> inferred from VGNet and STEP/B-rep structure.
- GNN architecture details -> required to implement encoder -> used BAGConv/VGNet lineage where CADGCL was vague.
- Exact feature definitions -> required fixed dimensions -> surface/curve schema defined from CADGCL, VGNet, and STEP observations.
- Training/inference engineering -> required working prototype -> implemented local pipeline, embeddings, vector retrieval, and UI.

### 33. Pause For Appreciation

Suggested wording:

`At this point, we were no longer only reproducing CADGCL. We were combining CADGCL with missing assumptions from related B-rep research to make a working prototype. This is exciting, but it also means uncertainty: we are operating at the edge of what the papers fully specify.`

### 34. FabWave Result: mAP@10 Compared With The Paper

Anchor this slide on the final `mAP@10` comparison between the CADGCL paper's FabWave result and the reproduced model.

Explain carefully:

- FabWave class labels are used as the available proxy for similarity.
- The comparison supports training quality and approximate paper parity.
- It should not be framed as perfect official reproduction because implementation assumptions and reproduction gaps remain.

Message: this is the strongest evidence that the model learned meaningful geometric embeddings before moving to Gühring data.

### 35. Example Pipeline: Cylinder As Raw Graph Tensors

Visualize exact data structures:

- `Cylinder STEP/B-rep`.
- `X: [num_faces x 16]`.
- `edge_index: [2 x num_edges]`.
- `edge_attr: [num_edges x 11]`.

Message: the CAD model enters the neural network as tensors, not as an image.

### 36. Tensor Operation: Message Passing

Visualize node rows updating through GNN layers:

`X0 -> X1 -> X2`

Each face vector receives information from neighboring face vectors plus edge attributes.

Message: local geometric relationships become richer learned node features.

### 37. Tensor Operation: Pooling

Visualize graph-level compression:

`[num_faces x hidden_dim] -> [1 x 256]`

Message: all learned face features are compressed into one vector for the whole CAD model.

### 38. Vector Operation: Similarity Search

Visualize:

`query vector -> compare against stored vectors -> nearest neighbors -> ranked CAD results`

Use cosine similarity or distance visually, without heavy formulas.

Message: similarity search becomes vector-distance comparison in the learned embedding space.

### 39. Prototype Setup For Live Demo

Static setup only:

`FabWave trained prototype -> local UI -> query model -> ranked results`

Closing line: `After this technical path, we can now see the model in action outside the slides.`

### 40. Why FabWave Was Not Enough

Message: FabWave proved the pipeline could train and retrieve similar geometry, but it does not fully represent Gühring's industrial complexity.

Supporting point: a working research prototype is not the same as a validated company solution.

### 41. New Dataset Requirements

The next dataset needed to be:

- Large enough for meaningful testing.
- Mechanically complex.
- Rich in similar or near-similar parts.
- Available as STEP or exportable CAD files.
- Paired with images for fast visual inspection.
- Close to the cutting-tool domain.

### 42. Datasets We Tried

Placeholder table for public datasets:

- Dataset name.
- Why considered.
- Why rejected or limited.

Message: finding a useful CAD similarity dataset was itself a major bottleneck.

### 43. Why Public Datasets Failed The Real Test

Common problems:

- Too simple.
- No meaningful near-duplicates.
- Weak or missing similarity labels.
- Poor image support.
- Categories do not represent functional similarity.

Message: public datasets helped development, but could not fully answer the Gühring business question.

### 44. Gühring Dataset: Real Industrial Complexity

Introduce the confidential internal dataset:

- Similia search result folders.
- STEP files.
- Screenshots.
- Excel files with Similia similarity scores.

Message: this is the first dataset that reflects the company's real search context.

### 45. Dataset Structure

Visualize one numbered folder as a search case:

`query/search set -> STEP files + screenshots + Excel similarity scores`

Message: each folder captures what Similia returned for a specific similarity search.

### 46. Dataset Statistics

Placeholder for extracted stats:

- Number of search folders.
- Number of STEP files.
- Number of images.
- Number of Similia rows/scores.
- Distribution of result-set sizes.

Message: define the scale of the internal benchmark.

### 47. Preparing The CADGCL vs Similia Comparison

Frame the possible evaluation paths because the exact comparison method depends on the next company meeting:

- Compare rankings for the same query parts.
- Inspect visual and functional quality of top-k results.
- Measure overlap between Similia and CADGCL.
- Ask Gühring experts which results are more reusable.

Message: the comparison must match how Gühring defines useful similarity.

### 48. Open Decision: How Should We Compare?

Present the decision to be made with the company:

- Numeric rank comparison.
- Expert preference review.
- Query-by-query case studies.
- Hybrid evaluation.

Message: before claiming improvement over Similia, the project needs a fair company-approved evaluation procedure.

### 49. From Prototype To Industrial Validation

Bridge into conclusions:

- CADGCL is trained and technically ready for internal testing.
- The remaining question is not only model performance, but whether its results are useful for Gühring engineers.
- The next company meeting defines how the final comparison and next steps should be presented.

## Deferred Content

The conclusion and next-step section will be designed after the company meeting, because the final comparison method and result interpretation are still open.

The seven starting papers, final bibliography count, public datasets tried, FabWave mAP@10 values, and Gühring dataset statistics should be inserted when the final data is available.

## HTML Implementation Notes

When implementation begins, create the slides as an HTML deck using the approved outline. The first version should prioritize structure, slide flow, and visual placeholders over final data completeness. Placeholder slides should clearly indicate which values or assets are still pending.

The deck should support a 1-hour presentation with the live demo performed outside the deck after slide 39.
