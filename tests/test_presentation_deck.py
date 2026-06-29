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


class PendingMarkerParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.unnamed_pending_markers = []
        self._placeholder_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if self._placeholder_depth or "data-placeholder-id" in attrs_dict:
            self._placeholder_depth += 1

    def handle_endtag(self, tag):
        if self._placeholder_depth:
            self._placeholder_depth -= 1

    def handle_data(self, data):
        if "Pending" in data and not self._placeholder_depth:
            self.unnamed_pending_markers.append(" ".join(data.split()))


def parse_deck():
    parser = DeckParser()
    parser.feed(DECK.read_text(encoding="utf-8"))
    return parser


def media_blocks(css):
    blocks = []
    for match in re.finditer(r"@media\s*\([^)]*max-width[^)]*\)\s*{", css):
        depth = 1
        index = match.end()
        while index < len(css) and depth:
            if css[index] == "{":
                depth += 1
            elif css[index] == "}":
                depth -= 1
            index += 1
        blocks.append(css[match.start():index])
    return blocks


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
            "The Business Cost Of Not Finding Parts",
            "What Similarity Means At Gühring",
            "Discovery Process",
            "Project Goals",
            "Selected Model: CADGCL",
            "Model Selection Criteria",
            "Prototype Setup For Live Demo",
            "Open Decision: How Should We Compare?",
            "From Prototype To Industrial Validation",
        ]
        for title in expected_titles:
            self.assertIn(title, titles)

    def test_introduction_slide_titles_match_visible_content(self):
        parser = parse_deck()
        titles_by_slide = {slide.get("data-slide"): slide.get("data-title") for slide in parser.slides}
        expected_titles = {
            "1": "Cover",
            "2": "Agenda",
            "3": "Why This Project Exists",
            "4": "The Business Cost Of Not Finding Parts",
            "5": "What Similarity Means At Gühring",
            "6": "Discovery Process",
            "7": "Project Goals",
        }
        self.assertEqual({slide: titles_by_slide[slide] for slide in expected_titles}, expected_titles)
        self.assertEqual(len(titles_by_slide.values()), len(set(titles_by_slide.values())))

    def test_intro_layouts_collapse_on_narrow_screens(self):
        css = CSS.read_text(encoding="utf-8")
        self.assertRegex(css, r"@media\s*\([^)]*max-width\s*:\s*[^)]*\)")
        for class_name in ["hero-grid", "two-column", "flow-row", "similarity-ladder"]:
            self.assertRegex(
                css,
                rf"@media[\s\S]*\.{class_name}[\s\S]*grid-template-columns\s*:\s*1fr",
            )

    def test_research_layouts_collapse_on_narrow_screens(self):
        css = CSS.read_text(encoding="utf-8")
        for class_name in ["comparison-grid", "criteria-grid"]:
            self.assertRegex(
                css,
                rf"@media[\s\S]*\.{class_name}[\s\S]*grid-template-columns\s*:\s*1fr",
            )

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
        placeholder_tags = re.findall(r"<[^>]*data-placeholder-id=\"[^\"]+\"[^>]*>", text)
        self.assertTrue(placeholder_tags, "Expected placeholder markers in deck markup")
        for tag in placeholder_tags:
            self.assertNotRegex(tag, r"\shidden(?:\s|=|>)")
        forbidden_claims = [
            "CADGCL outperformed Similia",
            "CADGCL is better than Similia",
            "final conclusion",
        ]
        for claim in forbidden_claims:
            self.assertNotIn(claim, text)

    def test_pending_markers_are_named_placeholders(self):
        parser = PendingMarkerParser()
        parser.feed(DECK.read_text(encoding="utf-8"))
        self.assertEqual(parser.unnamed_pending_markers, [])

    def test_cadgcl_architecture_layouts_collapse_on_narrow_screens(self):
        css = CSS.read_text(encoding="utf-8")
        mobile_blocks = media_blocks(css)
        for class_name in ["architecture-band", "pipeline-row", "tensor-grid", "hyperparameter-grid", "gap-table-row"]:
            self.assertTrue(
                any(
                    re.search(rf"\.{class_name}[\s\S]*?grid-template-columns\s*:\s*1fr", block)
                    for block in mobile_blocks
                ),
                f"Expected .{class_name} to collapse inside a max-width media query",
            )

    def test_slide_numbers_are_visible_in_markup(self):
        html = DECK.read_text(encoding="utf-8")
        matches = re.findall(r'class="slide-number">(\d+)/49</span>', html)
        self.assertEqual(matches, [str(i) for i in range(1, 50)])

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

    def test_seven_starting_papers_placeholder_is_table_structured(self):
        html = DECK.read_text(encoding="utf-8")
        slide_10 = re.search(
            r'<section class="slide" data-slide="10"[\s\S]*?</section>',
            html,
        ).group(0)
        self.assertIn('data-placeholder-id="PLACEHOLDER_SEVEN_STARTING_PAPERS"', slide_10)
        self.assertRegex(slide_10, r'<(?:table|div)[^>]*data-placeholder-id="PLACEHOLDER_SEVEN_STARTING_PAPERS"')
        for header in ["Paper", "Year", "Representation", "Relevance to Gühring"]:
            self.assertRegex(slide_10, rf"<th[^>]*>{header}</th>")

    def test_research_slide_titles_match_visible_content(self):
        parser = parse_deck()
        titles_by_slide = {slide.get("data-slide"): slide.get("data-title") for slide in parser.slides}
        self.assertEqual(titles_by_slide["17"], "Model Selection Criteria")

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

    def test_mobile_css_prevents_header_and_pending_table_overflow(self):
        css = CSS.read_text(encoding="utf-8")
        mobile_blocks = media_blocks(css)
        self.assertTrue(
            any(
                re.search(r"\.deck-header[\s\S]*?flex-wrap\s*:\s*wrap", block)
                and re.search(r"\.deck-controls[\s\S]*?flex-wrap\s*:\s*wrap", block)
                for block in mobile_blocks
            ),
            "Expected deck header and controls to wrap inside a max-width media query",
        )
        self.assertTrue(
            any(
                re.search(r"\.brand-lockup[\s\S]*?overflow-wrap\s*:\s*anywhere", block)
                and re.search(r"\.confidential[\s\S]*?overflow-wrap\s*:\s*anywhere", block)
                for block in mobile_blocks
            ),
            "Expected long header labels to break safely inside a max-width media query",
        )
        self.assertTrue(
            any(
                re.search(r"\.pending-table[\s\S]*?display\s*:\s*block", block)
                and re.search(r"\.pending-table[\s\S]*?overflow-x\s*:\s*auto", block)
                for block in mobile_blocks
            ),
            "Expected pending tables to scroll horizontally inside a max-width media query",
        )


if __name__ == "__main__":
    unittest.main()
