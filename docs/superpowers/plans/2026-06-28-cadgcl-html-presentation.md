# CADGCL HTML Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a branded HTML slide deck for the 1-hour internal Gühring CADGCL presentation through slide 49.

**Architecture:** Use a static, dependency-free HTML/CSS/JS deck under `presentation/`. Keep content in semantic slide sections, styling in one CSS file, and keyboard/navigation behavior in one JavaScript file. Add Python stdlib tests that verify slide count, required confidential markings, design tokens, and navigation hooks without needing browser automation.

**Tech Stack:** HTML5, CSS3, vanilla JavaScript, Python `unittest`/`html.parser` for validation.

## Global Constraints

- Deck is internal/confidential company material and may use Gühring dataset context, Similia material, and internal similarity definitions.
- Mark the deck as `Confidential | Internal Gühring Project`.
- Follow `Guhring DESIGN.md`: primary red `#CF2E2E`, secondary black `#1A1A1A`, white surfaces, structured industrial layouts, sans-serif typography, minimal rounding.
- Presentation structure is Decision Journey and company-focused; professor is present but not the primary audience.
- Technical depth is executive-technical balance: visually explain B-rep graphs, contrastive learning, BMM, implementation gaps, and evaluation with limited equations.
- Live demo happens outside the deck after slide 39; slide 39 only sets up the demo.
- Build slides through slide 49 only. Conclusions and next steps remain deferred until after the company meeting.
- Slides 47, 48, and 49 must remain explicitly undecided/to-be-decided and must not invent CADGCL-vs-Similia results.
- Pending data must be visibly marked as pending and must use stable placeholder IDs in the HTML. Every placeholder element must include `data-placeholder-id="PLACEHOLDER_NAME"` so the exact content can be filled later by name.

## Placeholder Registry

Use these exact placeholder IDs in the HTML deck. Do not create unnamed generic placeholders.

- `PLACEHOLDER_SEVEN_STARTING_PAPERS`: slide 10 table for the seven foundational papers.
- `PLACEHOLDER_FINAL_BIBLIOGRAPHY_COUNT`: slide 12 final number of reviewed papers.
- `PLACEHOLDER_FIELD_LIMITATION_REFERENCES`: slide 16 references for field limitations.
- `PLACEHOLDER_FABWAVE_MAP10_PAPER`: slide 34 CADGCL paper FabWave mAP@10 value.
- `PLACEHOLDER_FABWAVE_MAP10_REPRODUCTION`: slide 34 reproduced model FabWave mAP@10 value.
- `PLACEHOLDER_PUBLIC_DATASETS_TRIED`: slide 42 table of public datasets considered.
- `PLACEHOLDER_GUHRING_SEARCH_FOLDERS`: slide 46 number of Gühring search folders.
- `PLACEHOLDER_GUHRING_STEP_FILES`: slide 46 number of Gühring STEP files.
- `PLACEHOLDER_GUHRING_IMAGES`: slide 46 number of Gühring screenshots/images.
- `PLACEHOLDER_GUHRING_SIMILIA_SCORE_ROWS`: slide 46 number of Similia score rows.
- `PLACEHOLDER_GUHRING_RESULT_SET_DISTRIBUTION`: slide 46 distribution of result-set sizes.
- `PLACEHOLDER_SIMILIA_COMPARISON_PREPARATION`: slide 47 company-approved preparation text.
- `PLACEHOLDER_SIMILIA_COMPARISON_METHOD`: slide 48 company-approved comparison method.
- `PLACEHOLDER_INDUSTRIAL_VALIDATION_NEXT_STEP`: slide 49 company-approved validation bridge.

---

## File Structure

- Create `presentation/cadgcl-final-presentation.html`: the slide deck. One `<section class="slide" data-slide="N">` per slide, with `data-title` for testability and navigation.
- Create `presentation/styles.css`: Gühring-inspired visual system, slide layouts, progress indicator, cards, diagrams, tensor visuals, and print mode.
- Create `presentation/slides.js`: keyboard navigation, button navigation, slide counter, progress bar, and URL hash support.
- Create `tests/test_presentation_deck.py`: stdlib tests that parse the HTML and CSS to validate deck structure and required content.

---

### Task 1: Add Presentation Shell And Structural Tests

**Files:**
- Create: `presentation/cadgcl-final-presentation.html`
- Create: `presentation/styles.css`
- Create: `presentation/slides.js`
- Create: `tests/test_presentation_deck.py`

**Interfaces:**
- Produces: `presentation/cadgcl-final-presentation.html` with `<section class="slide" data-slide="1" data-title="Cover">` style slide sections.
- Produces: `presentation/styles.css` linked from the deck.
- Produces: `presentation/slides.js` linked from the deck.
- Produces: `tests/test_presentation_deck.py` with helper `DeckParser` and tests reused by later tasks.

- [ ] **Step 1: Write failing structural tests**

Create `tests/test_presentation_deck.py` with this complete content:

```python
from html.parser import HTMLParser
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "presentation" / "cadgcl-final-presentation.html"
CSS = ROOT / "presentation" / "styles.css"
JS = ROOT / "presentation" / "slides.js"


class DeckParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.slides = []
        self.links = []
        self.scripts = []
        self.text_parts = []
        self._capture_text = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "section" and "slide" in attrs_dict.get("class", "").split():
            self.slides.append(attrs_dict)
        if tag == "link":
            self.links.append(attrs_dict)
        if tag == "script":
            self.scripts.append(attrs_dict)
        if tag in {"h1", "h2", "h3", "p", "li", "span", "strong", "td", "th", "div"}:
            self._capture_text = True

    def handle_endtag(self, tag):
        if tag in {"h1", "h2", "h3", "p", "li", "span", "strong", "td", "th", "div"}:
            self._capture_text = False

    def handle_data(self, data):
        if self._capture_text:
            stripped = " ".join(data.split())
            if stripped:
                self.text_parts.append(stripped)


def parse_deck():
    parser = DeckParser()
    parser.feed(DECK.read_text(encoding="utf-8"))
    return parser


class PresentationDeckTests(unittest.TestCase):
    def test_deck_assets_exist_and_are_linked(self):
        self.assertTrue(DECK.exists(), "HTML deck should exist")
        self.assertTrue(CSS.exists(), "CSS file should exist")
        self.assertTrue(JS.exists(), "JS file should exist")
        parser = parse_deck()
        self.assertTrue(any(link.get("href") == "styles.css" for link in parser.links))
        self.assertTrue(any(script.get("src") == "slides.js" for script in parser.scripts))

    def test_deck_has_49_numbered_slides(self):
        parser = parse_deck()
        self.assertEqual(len(parser.slides), 49)
        self.assertEqual([slide.get("data-slide") for slide in parser.slides], [str(i) for i in range(1, 50)])
        self.assertTrue(all(slide.get("data-title") for slide in parser.slides))

    def test_deck_marks_confidential_internal_material(self):
        text = DECK.read_text(encoding="utf-8")
        self.assertGreaterEqual(text.count("Confidential | Internal Gühring Project"), 1)

    def test_css_contains_guhring_design_tokens(self):
        css = CSS.read_text(encoding="utf-8")
        self.assertIn("#CF2E2E", css)
        self.assertIn("#1A1A1A", css)
        self.assertIn("--radius-md: 4px", css)
        self.assertIn("font-family", css)

    def test_javascript_exposes_navigation_hooks(self):
        js = JS.read_text(encoding="utf-8")
        self.assertRegex(js, r"function\s+goToSlide")
        self.assertRegex(js, r"function\s+nextSlide")
        self.assertRegex(js, r"function\s+previousSlide")
        self.assertIn("keydown", js)

    def test_required_slide_titles_are_present(self):
        parser = parse_deck()
        titles = [slide.get("data-title") for slide in parser.slides]
        expected_titles = [
            "Cover",
            "Agenda",
            "What Similarity Means At Gühring",
            "Project Goals",
            "Initial Queries",
            "Selected Model: CADGCL",
            "FabWave Result: mAP@10 Compared With The Paper",
            "Prototype Setup For Live Demo",
            "Open Decision: How Should We Compare?",
            "From Prototype To Industrial Validation",
        ]
        for title in expected_titles:
            self.assertIn(title, titles)

    def test_pending_items_are_marked_without_fabricating_results(self):
        text = DECK.read_text(encoding="utf-8")
        self.assertIn("Pending", text)
        self.assertIn("to be decided", text)
        placeholder_ids = [
            "PLACEHOLDER_SEVEN_STARTING_PAPERS",
            "PLACEHOLDER_FINAL_BIBLIOGRAPHY_COUNT",
            "PLACEHOLDER_FIELD_LIMITATION_REFERENCES",
            "PLACEHOLDER_FABWAVE_MAP10_PAPER",
            "PLACEHOLDER_FABWAVE_MAP10_REPRODUCTION",
            "PLACEHOLDER_PUBLIC_DATASETS_TRIED",
            "PLACEHOLDER_GUHRING_SEARCH_FOLDERS",
            "PLACEHOLDER_GUHRING_STEP_FILES",
            "PLACEHOLDER_GUHRING_IMAGES",
            "PLACEHOLDER_GUHRING_SIMILIA_SCORE_ROWS",
            "PLACEHOLDER_GUHRING_RESULT_SET_DISTRIBUTION",
            "PLACEHOLDER_SIMILIA_COMPARISON_PREPARATION",
            "PLACEHOLDER_SIMILIA_COMPARISON_METHOD",
            "PLACEHOLDER_INDUSTRIAL_VALIDATION_NEXT_STEP",
        ]
        for placeholder_id in placeholder_ids:
            self.assertIn(f'data-placeholder-id="{placeholder_id}"', text)
        forbidden_claims = [
            "CADGCL outperformed Similia",
            "CADGCL is better than Similia",
            "final conclusion",
        ]
        for claim in forbidden_claims:
            self.assertNotIn(claim, text)

    def test_slide_numbers_are_visible_in_markup(self):
        html = DECK.read_text(encoding="utf-8")
        matches = re.findall(r'class="slide-number">(\d+)/49</span>', html)
        self.assertEqual(matches, [str(i) for i in range(1, 50)])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_presentation_deck -v`

Expected: FAIL because `presentation/cadgcl-final-presentation.html`, `presentation/styles.css`, and `presentation/slides.js` do not exist yet.

- [ ] **Step 3: Create the minimal deck shell with 49 numbered slides**

Create directory `presentation/` if it does not exist. Create `presentation/cadgcl-final-presentation.html` with a valid HTML document, linked `styles.css`, linked `slides.js`, and 49 slide sections. Use the exact slide titles from the spec as `data-title` values.

Required HTML skeleton:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CADGCL Final Presentation | Gühring Internal</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <div class="deck-shell">
    <header class="deck-header" aria-label="Presentation controls">
      <div class="brand-lockup"><span class="brand-mark"></span><span>Gühring CADGCL</span></div>
      <div class="confidential">Confidential | Internal Gühring Project</div>
      <div class="progress-track"><div class="progress-bar" id="progressBar"></div></div>
    </header>
    <main class="deck" id="deck" aria-live="polite">
      <!-- Add the 49 slide sections here in order. Each slide must include <span class="slide-number">N/49</span>. -->
    </main>
    <nav class="deck-controls" aria-label="Slide navigation">
      <button id="prevSlide" type="button">Previous</button>
      <span id="slideCounter">1 / 49</span>
      <button id="nextSlide" type="button">Next</button>
    </nav>
  </div>
  <script src="slides.js"></script>
</body>
</html>
```

For this task, each slide body may contain only its title and one sentence, but all 49 sections must exist. Use `to be decided` on slides 47, 48, and 49. Add all placeholder IDs from the Placeholder Registry as hidden or visible elements in their correct slide sections using `data-placeholder-id="PLACEHOLDER_NAME"`; later tasks will replace those elements with better visible layouts.

- [ ] **Step 4: Create CSS design tokens and minimal layout**

Create `presentation/styles.css` with this minimum content:

```css
:root {
  --guhring-red: #CF2E2E;
  --guhring-black: #1A1A1A;
  --surface: #FFFFFF;
  --text: #1A1A1A;
  --muted: #666666;
  --line: #E6E6E6;
  --radius-sm: 2px;
  --radius-md: 4px;
  --radius-lg: 8px;
  font-family: Arial, Helvetica, sans-serif;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: #F4F4F4;
  color: var(--text);
  font-family: Arial, Helvetica, sans-serif;
}

.deck-shell {
  min-height: 100vh;
  display: grid;
  grid-template-rows: auto 1fr auto;
}

.deck-header,
.deck-controls {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 12px 24px;
  background: var(--guhring-black);
  color: white;
}

.brand-lockup,
.confidential {
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.brand-mark {
  display: inline-block;
  width: 28px;
  height: 4px;
  margin-right: 10px;
  background: var(--guhring-red);
  vertical-align: middle;
}

.progress-track {
  width: 220px;
  height: 4px;
  background: rgba(255, 255, 255, 0.2);
}

.progress-bar {
  height: 100%;
  width: 0;
  background: var(--guhring-red);
}

.deck {
  position: relative;
  overflow: hidden;
}

.slide {
  display: none;
  min-height: calc(100vh - 104px);
  padding: 64px;
  background: var(--surface);
}

.slide.active { display: block; }

.slide-kicker {
  color: var(--guhring-red);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

h1, h2 {
  margin: 12px 0 24px;
  color: var(--guhring-black);
  font-weight: 700;
  line-height: 1.05;
}

h1 { font-size: clamp(42px, 6vw, 76px); }
h2 { font-size: clamp(34px, 4vw, 54px); }

p, li {
  font-size: clamp(18px, 2vw, 28px);
  line-height: 1.35;
}

.slide-number {
  position: absolute;
  right: 32px;
  bottom: 28px;
  color: var(--muted);
  font-size: 13px;
  font-weight: 600;
}

button {
  border: 0;
  border-radius: var(--radius-md);
  padding: 10px 16px;
  background: var(--guhring-red);
  color: white;
  font-weight: 700;
  text-transform: uppercase;
}

@media print {
  .deck-header, .deck-controls { display: none; }
  .slide { display: block; min-height: 100vh; page-break-after: always; }
}
```

- [ ] **Step 5: Create navigation JavaScript**

Create `presentation/slides.js` with this complete content:

```javascript
const slides = Array.from(document.querySelectorAll('.slide'));
const progressBar = document.getElementById('progressBar');
const slideCounter = document.getElementById('slideCounter');
let currentSlide = 0;

function clampSlide(index) {
  return Math.max(0, Math.min(index, slides.length - 1));
}

function updateUrl(index) {
  const slideNumber = index + 1;
  if (window.location.hash !== `#${slideNumber}`) {
    window.history.replaceState(null, '', `#${slideNumber}`);
  }
}

function goToSlide(index) {
  currentSlide = clampSlide(index);
  slides.forEach((slide, slideIndex) => {
    slide.classList.toggle('active', slideIndex === currentSlide);
    slide.setAttribute('aria-hidden', slideIndex === currentSlide ? 'false' : 'true');
  });
  const slideNumber = currentSlide + 1;
  if (slideCounter) slideCounter.textContent = `${slideNumber} / ${slides.length}`;
  if (progressBar) progressBar.style.width = `${(slideNumber / slides.length) * 100}%`;
  updateUrl(currentSlide);
}

function nextSlide() {
  goToSlide(currentSlide + 1);
}

function previousSlide() {
  goToSlide(currentSlide - 1);
}

document.addEventListener('keydown', (event) => {
  if (event.key === 'ArrowRight' || event.key === 'PageDown' || event.key === ' ') {
    event.preventDefault();
    nextSlide();
  }
  if (event.key === 'ArrowLeft' || event.key === 'PageUp') {
    event.preventDefault();
    previousSlide();
  }
  if (event.key === 'Home') {
    event.preventDefault();
    goToSlide(0);
  }
  if (event.key === 'End') {
    event.preventDefault();
    goToSlide(slides.length - 1);
  }
});

document.getElementById('nextSlide')?.addEventListener('click', nextSlide);
document.getElementById('prevSlide')?.addEventListener('click', previousSlide);

const hashSlide = Number.parseInt(window.location.hash.replace('#', ''), 10);
goToSlide(Number.isFinite(hashSlide) ? hashSlide - 1 : 0);
```

- [ ] **Step 6: Run structural tests**

Run: `python3 -m unittest tests.test_presentation_deck -v`

Expected: PASS.

- [ ] **Step 7: Commit shell and tests**

Run:

```bash
git add presentation/cadgcl-final-presentation.html presentation/styles.css presentation/slides.js tests/test_presentation_deck.py
git commit -m "feat: add CADGCL presentation shell"
```

Expected: commit succeeds with only these four files staged.

---

### Task 2: Implement Introduction Slides 1-7

**Files:**
- Modify: `presentation/cadgcl-final-presentation.html`
- Modify: `presentation/styles.css`
- Test: `tests/test_presentation_deck.py`

**Interfaces:**
- Consumes: 49 slide shell from Task 1.
- Produces: complete introduction content for slides 1-7, using company-defined similarity from `Similarity.pdf`.

- [ ] **Step 1: Add tests for introduction content**

Append this test method to `PresentationDeckTests` in `tests/test_presentation_deck.py`:

```python
    def test_introduction_slides_contain_company_problem_and_similarity_definition(self):
        text = " ".join(parse_deck().text_parts)
        required_phrases = [
            "AI-Based Geometric Similarity Search for 3D CAD Models",
            "670,000+ CAD models",
            "Similia results are not accurate enough",
            "design from scratch",
            "Full Similarity",
            "Functional Similarity",
            "Contour Similarity",
            "size-invariant",
            "Can current AI research provide a viable solution",
        ]
        for phrase in required_phrases:
            self.assertIn(phrase, text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_presentation_deck.PresentationDeckTests.test_introduction_slides_contain_company_problem_and_similarity_definition -v`

Expected: FAIL because slide content is still minimal.

- [ ] **Step 3: Replace slides 1-7 with final content**

In `presentation/cadgcl-final-presentation.html`, replace the inner content of sections `data-slide="1"` through `data-slide="7"` with concise, presentation-ready content from the spec:

- Slide 1: Cover title, subtitle, confidential marker.
- Slide 2: Agenda with five sections.
- Slide 3: Gühring has `670,000+ CAD models`; barriers are fragmented search, inconsistent metadata, local templates, knowledge barriers, and Similia results not accurate enough.
- Slide 4: decision flow `find existing design -> adapt proven component` and `fail to find -> design from scratch`; business impact list.
- Slide 5: company similarity definition: `Full Similarity -> Functional Similarity -> Contour Similarity`; size-invariant comparison.
- Slide 6: discovery process findings.
- Slide 7: project goals and main question.

Use reusable classes already in CSS plus these new classes where helpful: `hero-grid`, `agenda-list`, `metric-card`, `flow-row`, `similarity-ladder`, `goal-panel`.

- [ ] **Step 4: Add introduction layout CSS**

Append these styles to `presentation/styles.css`:

```css
.hero-grid,
.two-column {
  display: grid;
  grid-template-columns: 1.1fr 0.9fr;
  gap: 40px;
  align-items: center;
}

.hero-panel,
.metric-card,
.goal-panel,
.decision-card {
  border-left: 6px solid var(--guhring-red);
  border-radius: var(--radius-md);
  padding: 28px;
  background: #F7F7F7;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
}

.agenda-list,
.impact-list,
.finding-list {
  display: grid;
  gap: 14px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.agenda-list li,
.impact-list li,
.finding-list li {
  border-bottom: 1px solid var(--line);
  padding: 14px 0;
}

.flow-row {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 20px;
  align-items: center;
}

.flow-arrow {
  color: var(--guhring-red);
  font-size: 44px;
  font-weight: 700;
}

.similarity-ladder {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 18px;
}

.similarity-card {
  border-top: 8px solid var(--guhring-red);
  border-radius: var(--radius-md);
  padding: 24px;
  background: #F7F7F7;
}
```

- [ ] **Step 5: Run full presentation tests**

Run: `python3 -m unittest tests.test_presentation_deck -v`

Expected: PASS.

- [ ] **Step 6: Commit introduction slides**

Run:

```bash
git add presentation/cadgcl-final-presentation.html presentation/styles.css tests/test_presentation_deck.py
git commit -m "feat: add presentation introduction slides"
```

Expected: commit succeeds with only these three files staged.

---

### Task 3: Implement Research And Model Selection Slides 8-19

**Files:**
- Modify: `presentation/cadgcl-final-presentation.html`
- Modify: `presentation/styles.css`
- Test: `tests/test_presentation_deck.py`

**Interfaces:**
- Consumes: deck shell and introduction slides.
- Produces: research funnel, real initial queries, bibliography placeholders, model-selection criteria, and CADGCL selection slide.

- [ ] **Step 1: Add tests for research content**

Append this test method to `PresentationDeckTests`:

```python
    def test_research_slides_contain_real_queries_and_model_selection_criteria(self):
        text = " ".join(parse_deck().text_parts)
        required_phrases = [
            "2020 onward",
            "AI CAD similarity search",
            "CAD geometric similarity search",
            "ML geometric similarity search",
            "Research Rabbit",
            "Final bibliography",
            "From Voxels To B-rep",
            "UV-Net",
            "B-rep support",
            "Unsupervised capability",
            "Direct measured superiority over other models",
            "Number of benchmark comparison",
            "BRepMAE",
            "CADGCL",
        ]
        for phrase in required_phrases:
            self.assertIn(phrase, text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_presentation_deck.PresentationDeckTests.test_research_slides_contain_real_queries_and_model_selection_criteria -v`

Expected: FAIL because research slide content is still minimal.

- [ ] **Step 3: Replace slides 8-19 with final research content**

Update the relevant slide sections:

- Slide 8: research strategy and 2020 onward bias.
- Slide 9: exact initial queries from `Poster Guhring.pdf`.
- Slide 10: pending seven starting papers table with visible `Pending` labels and `data-placeholder-id="PLACEHOLDER_SEVEN_STARTING_PAPERS"`.
- Slide 11: Research Rabbit citation expansion.
- Slide 12: final bibliography count as a large pending number, not grouped by theme, with `data-placeholder-id="PLACEHOLDER_FINAL_BIBLIOGRAPHY_COUNT"`.
- Slide 13: literature pattern from voxels/point clouds/multi-view/mesh to B-rep.
- Slide 14: why B-rep won compared to voxel, point cloud, multi-view, mesh.
- Slide 15: UV-Net as seminal reference.
- Slide 16: field limitations and pending references with `data-placeholder-id="PLACEHOLDER_FIELD_LIMITATION_REFERENCES"`.
- Slide 17: model selection criteria exactly as updated by the user: exclusion criteria `B-rep support`, `Unsupervised capability`; ranking criteria `Direct measured superiority over other models`, `Number of benchmark comparison`.
- Slide 18: BRepMAE vs CADGCL.
- Slide 19: selected model CADGCL, without naming it `Decision 2`.

- [ ] **Step 4: Add research layout CSS**

Append these styles to `presentation/styles.css`:

```css
.query-stack,
.criteria-grid,
.comparison-grid,
.timeline {
  display: grid;
  gap: 18px;
}

.query-pill,
.timeline-item,
.criteria-card,
.comparison-card,
.pending-card {
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  padding: 22px;
  background: #FFFFFF;
}

.query-pill {
  border-left: 6px solid var(--guhring-red);
  font-weight: 700;
}

.comparison-grid,
.criteria-grid {
  grid-template-columns: repeat(2, 1fr);
}

.pending-badge {
  display: inline-block;
  border-radius: var(--radius-sm);
  padding: 6px 10px;
  background: var(--guhring-black);
  color: white;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.big-number {
  color: var(--guhring-red);
  font-size: clamp(80px, 14vw, 180px);
  font-weight: 800;
  line-height: 0.9;
}
```

- [ ] **Step 5: Run full presentation tests**

Run: `python3 -m unittest tests.test_presentation_deck -v`

Expected: PASS.

- [ ] **Step 6: Commit research slides**

Run:

```bash
git add presentation/cadgcl-final-presentation.html presentation/styles.css tests/test_presentation_deck.py
git commit -m "feat: add research selection slides"
```

Expected: commit succeeds with only these three files staged.

---

### Task 4: Implement CADGCL Architecture Slides 20-34

**Files:**
- Modify: `presentation/cadgcl-final-presentation.html`
- Modify: `presentation/styles.css`
- Test: `tests/test_presentation_deck.py`

**Interfaces:**
- Consumes: deck with slides 1-19 complete.
- Produces: executive-technical CADGCL architecture section through FabWave mAP@10 comparison placeholder.

- [ ] **Step 1: Add tests for CADGCL architecture content**

Append this test method to `PresentationDeckTests`:

```python
    def test_cadgcl_architecture_slides_cover_training_pipeline_and_reproduction_gaps(self):
        text = " ".join(parse_deck().text_parts)
        required_phrases = [
            "state-of-the-art research territory",
            "VGNet",
            "Data transformation and augmentation",
            "Beta Mixture Model",
            "Faces/surfaces become nodes",
            "Node matrix X",
            "edge_index",
            "edge_attr",
            "G_original",
            "z_original",
            "G_augmented",
            "z_augmented",
            "false negative",
            "post-training embeddings become the searchable database",
            "Epochs: 20",
            "Batch size: 128",
            "mAP@10",
        ]
        for phrase in required_phrases:
            self.assertIn(phrase, text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_presentation_deck.PresentationDeckTests.test_cadgcl_architecture_slides_cover_training_pipeline_and_reproduction_gaps -v`

Expected: FAIL because CADGCL slide content is still minimal.

- [ ] **Step 3: Replace slides 20-34 with final CADGCL content**

Update slide sections 20-34 exactly according to the spec:

- Slide 20: two disclaimers.
- Slide 21: CADGCL missing details reconstructed through VGNet.
- Slide 22: three CADGCL parts.
- Slide 23: B-rep to graph.
- Slide 24: node matrix `X`, `edge_index`, `edge_attr`.
- Slide 25: original graph and augmented graph.
- Slide 26: `G_original -> shared GNN -> z_original` and `G_augmented -> same shared GNN -> z_augmented`.
- Slide 27: contrastive learning.
- Slide 28: false negative problem.
- Slide 29: BMM negative sampling.
- Slide 30: training embeddings vs post-training searchable embeddings.
- Slide 31: hyperparameters.
- Slide 32: missing paper details table.
- Slide 33: pause for appreciation.
- Slide 34: FabWave mAP@10 comparison, with values visibly marked `Pending` until final numbers are inserted. The paper value must use `data-placeholder-id="PLACEHOLDER_FABWAVE_MAP10_PAPER"`; the reproduced model value must use `data-placeholder-id="PLACEHOLDER_FABWAVE_MAP10_REPRODUCTION"`.

- [ ] **Step 4: Add CADGCL architecture CSS**

Append these styles to `presentation/styles.css`:

```css
.architecture-band,
.pipeline-row,
.tensor-grid,
.hyperparameter-grid,
.gap-table {
  display: grid;
  gap: 18px;
}

.architecture-band {
  grid-template-columns: repeat(3, 1fr);
}

.pipeline-row {
  grid-template-columns: repeat(5, minmax(0, 1fr));
  align-items: stretch;
}

.pipeline-step,
.tensor-box,
.hyperparameter,
.gap-table-row {
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  padding: 20px;
  background: #F7F7F7;
}

.tensor-box code,
.pipeline-step code {
  color: var(--guhring-red);
  font-weight: 700;
}

.hyperparameter-grid {
  grid-template-columns: repeat(4, 1fr);
}

.hyperparameter strong {
  display: block;
  color: var(--guhring-red);
  font-size: 30px;
}

.gap-table-row {
  display: grid;
  grid-template-columns: 1fr 1fr 1.2fr;
  gap: 16px;
}
```

- [ ] **Step 5: Run full presentation tests**

Run: `python3 -m unittest tests.test_presentation_deck -v`

Expected: PASS.

- [ ] **Step 6: Commit CADGCL architecture slides**

Run:

```bash
git add presentation/cadgcl-final-presentation.html presentation/styles.css tests/test_presentation_deck.py
git commit -m "feat: add CADGCL architecture slides"
```

Expected: commit succeeds with only these three files staged.

---

### Task 5: Implement Tensor Walkthrough And Demo Setup Slides 35-39

**Files:**
- Modify: `presentation/cadgcl-final-presentation.html`
- Modify: `presentation/styles.css`
- Test: `tests/test_presentation_deck.py`

**Interfaces:**
- Consumes: architecture section through slide 34.
- Produces: visual tensor/math walkthrough and live-demo setup slide.

- [ ] **Step 1: Add tests for tensor walkthrough content**

Append this test method to `PresentationDeckTests`:

```python
    def test_tensor_walkthrough_slides_visualize_graph_tensor_vector_operations(self):
        text = " ".join(parse_deck().text_parts)
        required_phrases = [
            "Cylinder STEP/B-rep",
            "X: [num_faces x 16]",
            "edge_index: [2 x num_edges]",
            "edge_attr: [num_edges x 11]",
            "X0 -> X1 -> X2",
            "[num_faces x hidden_dim] -> [1 x 256]",
            "query vector",
            "nearest neighbors",
            "live demo",
        ]
        for phrase in required_phrases:
            self.assertIn(phrase, text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_presentation_deck.PresentationDeckTests.test_tensor_walkthrough_slides_visualize_graph_tensor_vector_operations -v`

Expected: FAIL because tensor walkthrough slide content is still minimal.

- [ ] **Step 3: Replace slides 35-39 with visual tensor walkthrough content**

Update slide sections 35-39:

- Slide 35: cylinder as raw graph tensors: `Cylinder STEP/B-rep`, `X: [num_faces x 16]`, `edge_index: [2 x num_edges]`, `edge_attr: [num_edges x 11]`.
- Slide 36: message passing: `X0 -> X1 -> X2`; face vectors receive neighbor and edge information.
- Slide 37: pooling: `[num_faces x hidden_dim] -> [1 x 256]`.
- Slide 38: vector search: query vector compared against stored vectors, nearest neighbors returned.
- Slide 39: prototype setup for live demo only, not the demo itself.

- [ ] **Step 4: Add tensor visualization CSS**

Append these styles to `presentation/styles.css`:

```css
.matrix-visual {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 6px;
  padding: 18px;
  border-radius: var(--radius-md);
  background: var(--guhring-black);
}

.matrix-cell {
  min-height: 32px;
  border-radius: var(--radius-sm);
  background: white;
}

.matrix-cell.accent {
  background: var(--guhring-red);
}

.vector-strip {
  display: flex;
  gap: 6px;
  align-items: center;
}

.vector-cell {
  width: 18px;
  height: 72px;
  border-radius: var(--radius-sm);
  background: var(--guhring-red);
}

.embedding-space {
  position: relative;
  min-height: 320px;
  border: 1px solid var(--line);
  background: linear-gradient(135deg, #FFFFFF 0%, #F2F2F2 100%);
}

.point {
  position: absolute;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--guhring-red);
}

.point.secondary {
  background: var(--guhring-black);
}
```

- [ ] **Step 5: Run full presentation tests**

Run: `python3 -m unittest tests.test_presentation_deck -v`

Expected: PASS.

- [ ] **Step 6: Commit tensor walkthrough slides**

Run:

```bash
git add presentation/cadgcl-final-presentation.html presentation/styles.css tests/test_presentation_deck.py
git commit -m "feat: add CADGCL tensor walkthrough slides"
```

Expected: commit succeeds with only these three files staged.

---

### Task 6: Implement Scaling Complexity Slides 40-49

**Files:**
- Modify: `presentation/cadgcl-final-presentation.html`
- Modify: `presentation/styles.css`
- Test: `tests/test_presentation_deck.py`

**Interfaces:**
- Consumes: complete deck through slide 39.
- Produces: scaling complexity section with Gühring dataset context and undecided comparison slides.

- [ ] **Step 1: Add tests for scaling complexity content**

Append this test method to `PresentationDeckTests`:

```python
    def test_scaling_complexity_slides_stop_before_unapproved_similia_comparison(self):
        text = " ".join(parse_deck().text_parts)
        required_phrases = [
            "Why FabWave Was Not Enough",
            "mechanically complex",
            "Datasets We Tried",
            "Why Public Datasets Failed The Real Test",
            "Gühring Dataset: Real Industrial Complexity",
            "Similia search result folders",
            "Excel files with Similia similarity scores",
            "Dataset Statistics",
            "Preparing The CADGCL vs Similia Comparison",
            "Open Decision: How Should We Compare?",
            "From Prototype To Industrial Validation",
            "to be decided",
        ]
        for phrase in required_phrases:
            self.assertIn(phrase, text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_presentation_deck.PresentationDeckTests.test_scaling_complexity_slides_stop_before_unapproved_similia_comparison -v`

Expected: FAIL because scaling slide content is still minimal.

- [ ] **Step 3: Replace slides 40-49 with scaling complexity content**

Update slide sections 40-49:

- Slide 40: FabWave was not enough.
- Slide 41: new dataset requirements.
- Slide 42: datasets tried table with `Pending` rows and `data-placeholder-id="PLACEHOLDER_PUBLIC_DATASETS_TRIED"`.
- Slide 43: why public datasets failed the real test.
- Slide 44: Gühring dataset as real industrial complexity.
- Slide 45: dataset structure: `query/search set -> STEP files + screenshots + Excel similarity scores`.
- Slide 46: dataset statistics with all values visibly `Pending`. Use one placeholder element per statistic: `PLACEHOLDER_GUHRING_SEARCH_FOLDERS`, `PLACEHOLDER_GUHRING_STEP_FILES`, `PLACEHOLDER_GUHRING_IMAGES`, `PLACEHOLDER_GUHRING_SIMILIA_SCORE_ROWS`, and `PLACEHOLDER_GUHRING_RESULT_SET_DISTRIBUTION`.
- Slide 47: preparing CADGCL vs Similia comparison; body text must include `to be decided` and `data-placeholder-id="PLACEHOLDER_SIMILIA_COMPARISON_PREPARATION"`.
- Slide 48: open decision: how should we compare; body text must include `to be decided` and `data-placeholder-id="PLACEHOLDER_SIMILIA_COMPARISON_METHOD"`.
- Slide 49: from prototype to industrial validation; body text must include `to be decided`, `data-placeholder-id="PLACEHOLDER_INDUSTRIAL_VALIDATION_NEXT_STEP"`, and must not claim results.

- [ ] **Step 4: Add scaling section CSS**

Append these styles to `presentation/styles.css`:

```css
.dataset-table,
.stats-grid,
.decision-options {
  display: grid;
  gap: 16px;
}

.dataset-table-row {
  display: grid;
  grid-template-columns: 1fr 1.2fr 1.4fr;
  gap: 16px;
  border-bottom: 1px solid var(--line);
  padding: 14px 0;
}

.stats-grid {
  grid-template-columns: repeat(5, 1fr);
}

.stat-card {
  border-radius: var(--radius-md);
  padding: 20px;
  background: var(--guhring-black);
  color: white;
}

.stat-card strong {
  display: block;
  color: var(--guhring-red);
  font-size: 34px;
}

.decision-options {
  grid-template-columns: repeat(4, 1fr);
}

.to-be-decided {
  border: 2px dashed var(--guhring-red);
  border-radius: var(--radius-md);
  padding: 24px;
  background: #FFF7F7;
  color: var(--guhring-black);
  font-weight: 700;
}
```

- [ ] **Step 5: Run full presentation tests**

Run: `python3 -m unittest tests.test_presentation_deck -v`

Expected: PASS.

- [ ] **Step 6: Commit scaling complexity slides**

Run:

```bash
git add presentation/cadgcl-final-presentation.html presentation/styles.css tests/test_presentation_deck.py
git commit -m "feat: add scaling complexity slides"
```

Expected: commit succeeds with only these three files staged.

---

### Task 7: Polish Responsive Presentation Experience

**Files:**
- Modify: `presentation/cadgcl-final-presentation.html`
- Modify: `presentation/styles.css`
- Modify: `presentation/slides.js`
- Test: `tests/test_presentation_deck.py`

**Interfaces:**
- Consumes: complete content through slide 49.
- Produces: polished deck with accessible navigation, mobile fallback, print behavior, and visual consistency.

- [ ] **Step 1: Add polish tests**

Append this test method to `PresentationDeckTests`:

```python
    def test_deck_has_accessibility_and_responsive_features(self):
        html = DECK.read_text(encoding="utf-8")
        css = CSS.read_text(encoding="utf-8")
        js = JS.read_text(encoding="utf-8")
        self.assertIn('aria-label="Presentation controls"', html)
        self.assertIn('aria-label="Slide navigation"', html)
        self.assertIn('@media print', css)
        self.assertIn('@media (max-width: 900px)', css)
        self.assertIn("aria-hidden", js)
        self.assertIn("replaceState", js)
```

- [ ] **Step 2: Run test to verify it fails if responsive CSS is missing**

Run: `python3 -m unittest tests.test_presentation_deck.PresentationDeckTests.test_deck_has_accessibility_and_responsive_features -v`

Expected: FAIL until `@media (max-width: 900px)` is added.

- [ ] **Step 3: Add responsive CSS**

Append this block to `presentation/styles.css`:

```css
@media (max-width: 900px) {
  .deck-header,
  .deck-controls {
    padding: 10px 14px;
  }

  .progress-track {
    display: none;
  }

  .slide {
    min-height: calc(100vh - 96px);
    padding: 32px 22px 64px;
  }

  .hero-grid,
  .two-column,
  .similarity-ladder,
  .comparison-grid,
  .criteria-grid,
  .architecture-band,
  .pipeline-row,
  .hyperparameter-grid,
  .stats-grid,
  .decision-options {
    grid-template-columns: 1fr;
  }

  .gap-table-row,
  .dataset-table-row {
    grid-template-columns: 1fr;
  }

  p, li {
    font-size: 18px;
  }
}
```

- [ ] **Step 4: Manually inspect the deck in a browser**

Run: `python3 -m http.server 8000 --directory presentation`

Expected: server starts and serves the deck at `http://localhost:8000/cadgcl-final-presentation.html`.

Manual checks:

- Arrow keys move between slides.
- Previous/Next buttons work.
- Slide counter updates.
- URL hash updates.
- First slide and slide 39 are readable on desktop width.
- Slides remain readable below 900px width.

Stop the server with `Ctrl+C` after inspection.

- [ ] **Step 5: Run full presentation tests**

Run: `python3 -m unittest tests.test_presentation_deck -v`

Expected: PASS.

- [ ] **Step 6: Commit presentation polish**

Run:

```bash
git add presentation/cadgcl-final-presentation.html presentation/styles.css presentation/slides.js tests/test_presentation_deck.py
git commit -m "feat: polish CADGCL presentation deck"
```

Expected: commit succeeds with only these four files staged.

---

### Task 8: Final Verification And Handoff Notes

**Files:**
- Create: `presentation/README.md`
- Test: `tests/test_presentation_deck.py`

**Interfaces:**
- Consumes: complete HTML deck.
- Produces: user-facing instructions for opening the deck and updating pending content later.

- [ ] **Step 1: Add README test**

Append this test method to `PresentationDeckTests`:

```python
    def test_presentation_readme_explains_how_to_open_and_update_pending_items(self):
        readme = ROOT / "presentation" / "README.md"
        self.assertTrue(readme.exists())
        text = readme.read_text(encoding="utf-8")
        self.assertIn("cadgcl-final-presentation.html", text)
        self.assertIn("python3 -m http.server 8000 --directory presentation", text)
        self.assertIn("Pending content", text)
        self.assertIn("Placeholder IDs", text)
        self.assertIn("PLACEHOLDER_SEVEN_STARTING_PAPERS", text)
        self.assertIn("PLACEHOLDER_SIMILIA_COMPARISON_METHOD", text)
        self.assertIn("Confidential", text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_presentation_deck.PresentationDeckTests.test_presentation_readme_explains_how_to_open_and_update_pending_items -v`

Expected: FAIL because `presentation/README.md` does not exist yet.

- [ ] **Step 3: Create presentation README**

Create `presentation/README.md` with this content:

```markdown
# CADGCL Final Presentation

Confidential internal Gühring project material.

## Open The Deck

Run:

```bash
python3 -m http.server 8000 --directory presentation
```

Then open:

`http://localhost:8000/cadgcl-final-presentation.html`

The deck also opens directly as a local file in modern browsers, but the local server path is preferred.

## Navigation

- Right arrow, Page Down, or Space: next slide.
- Left arrow or Page Up: previous slide.
- Home: first slide.
- End: last slide.
- Previous/Next buttons are available at the bottom of the deck.

## Pending content

The following items are intentionally marked pending until the final project data is available:

- `PLACEHOLDER_SEVEN_STARTING_PAPERS`: seven starting research papers.
- `PLACEHOLDER_FINAL_BIBLIOGRAPHY_COUNT`: final bibliography count.
- `PLACEHOLDER_FIELD_LIMITATION_REFERENCES`: field limitation references.
- `PLACEHOLDER_FABWAVE_MAP10_PAPER`: CADGCL paper FabWave mAP@10.
- `PLACEHOLDER_FABWAVE_MAP10_REPRODUCTION`: reproduced model FabWave mAP@10.
- `PLACEHOLDER_PUBLIC_DATASETS_TRIED`: public datasets tried.
- `PLACEHOLDER_GUHRING_SEARCH_FOLDERS`: number of Gühring search folders.
- `PLACEHOLDER_GUHRING_STEP_FILES`: number of Gühring STEP files.
- `PLACEHOLDER_GUHRING_IMAGES`: number of Gühring screenshots/images.
- `PLACEHOLDER_GUHRING_SIMILIA_SCORE_ROWS`: number of Similia score rows.
- `PLACEHOLDER_GUHRING_RESULT_SET_DISTRIBUTION`: distribution of result-set sizes.
- `PLACEHOLDER_SIMILIA_COMPARISON_PREPARATION`: slide 47 comparison preparation.
- `PLACEHOLDER_SIMILIA_COMPARISON_METHOD`: slide 48 comparison method.
- `PLACEHOLDER_INDUSTRIAL_VALIDATION_NEXT_STEP`: slide 49 validation bridge.

## Placeholder IDs

Every pending element in `cadgcl-final-presentation.html` has a `data-placeholder-id` attribute. When providing replacement files later, name them after these IDs so each asset or text block maps directly to one slot.

Do not replace slides 47-49 with results until the company-approved comparison method is defined.
```

- [ ] **Step 4: Run all tests**

Run: `python3 -m unittest tests.test_presentation_deck -v`

Expected: PASS.

- [ ] **Step 5: Inspect final git diff**

Run: `git diff -- presentation tests/test_presentation_deck.py`

Expected: diff includes only the presentation deck, presentation assets, README, and presentation test file.

- [ ] **Step 6: Commit final handoff docs**

Run:

```bash
git add presentation/README.md tests/test_presentation_deck.py
git commit -m "docs: add presentation handoff notes"
```

Expected: commit succeeds with only these two files staged.

---

## Self-Review

Spec coverage:

- Slides 1-49 are covered by Tasks 2-6.
- Gühring visual system is covered by Tasks 1, 2, 3, 4, 5, 6, and 7.
- Confidential marking is covered by Task 1 and tests.
- Real initial queries are covered by Task 3.
- Updated model-selection criteria are covered by Task 3.
- Correct original/augmented embedding explanation is covered by Task 4.
- Post-training embedding/search infrastructure explanation is covered by Task 4.
- Visual tensor/vector walkthrough is covered by Task 5.
- Deferred Similia comparison and to-be-decided slides are covered by Task 6.
- Handoff/update instructions are covered by Task 8.

Placeholder scan:

- The plan contains intentional `Pending` and `to be decided` content only where the spec requires pending data or undecided slides.
- The plan does not instruct workers to invent final values or conclusions.

Type/interface consistency:

- Deck path is consistently `presentation/cadgcl-final-presentation.html`.
- CSS path is consistently `presentation/styles.css`.
- JS path is consistently `presentation/slides.js`.
- Test helper names are consistent: `DeckParser`, `parse_deck`, `PresentationDeckTests`.
