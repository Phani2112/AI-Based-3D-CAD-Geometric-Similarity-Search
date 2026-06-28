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
            "The Business Cost Of Not Finding Parts",
            "What Similarity Means At Gühring",
            "Discovery Process",
            "Project Goals",
            "Selected Model: CADGCL",
            "FabWave Result: mAP@10 Compared With The Paper",
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


if __name__ == "__main__":
    unittest.main()
