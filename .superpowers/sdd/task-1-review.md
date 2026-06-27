## Task 1 Review: STEP Parser Module

### Diff (from commit eb0e74e)

```diff
diff --git a/src/data/step_parser.py b/src/data/step_parser.py
--- /dev/null
+++ b/src/data/step_parser.py
@@ -0,0 +1,26 @@
+from OCC.Core.STEPControl import STEPControl_Reader
+from OCC.Core.IFSelect import IFSelect_RetDone
+from OCC.Core.TopExp import TopExp_Explorer
+from OCC.Core.TopAbs import TopAbs_FACE
+from OCC.Core.TopoDS import topods
+
+class STEPParser:
+    def parse(self, filepath):
+        reader = STEPControl_Reader()
+        status = reader.ReadFile(filepath)
+        if status == IFSelect_RetDone:
+            reader.TransferRoots()
+            shape = reader.OneShape()
+            if not shape.IsNull():
+                return ParsedShape(shape)
+        return None
+
+class ParsedShape:
+    def __init__(self, shape):
+        self._shape = shape
+        self.Faces = []
+        exp = TopExp_Explorer(shape, TopAbs_FACE)
+        while exp.More():
+            face = topods.Face(exp.Current())
+            self.Faces.append(face)
+            exp.Next()
\ No newline at end of file

diff --git a/tests/test_step_parser.py b/tests/test_step_parser.py
--- /dev/null
+++ b/tests/test_step_parser.py
@@ -0,0 +1,9 @@
+import pytest
+from src.data.step_parser import STEPParser
+
+def test_parse_step_file():
+    parser = STEPParser()
+    shape = parser.parse("tests/data/sample.stp")
+    assert shape is not None
+    assert hasattr(shape, 'Faces')
+    assert len(shape.Faces) > 0
\ No newline at end of file
```

### SPEC: ✅ COMPLIANT

The implementation meets all requirements from the brief:
- Created `src/data/step_parser.py` with `STEPParser` class
- Created `tests/test_step_parser.py` with the specified test
- `STEPParser.parse()` returns an object with `Faces` attribute (via `ParsedShape` wrapper)
- Sample STEP file exists at `tests/data/sample.stp`

### QUALITY: ⚠️ NEEDS WORK

**Strengths:**
- Implementation extends the spec by adding `ParsedShape` wrapper class for cleaner interface
- Null shape check prevents returning empty shapes
- Proper extraction of faces using `TopExp_Explorer`
- Test file matches the specified test case

**Issues:**
- 🔴 CRITICAL: Missing dependency - `pythonocc-core` (OCC) is not installed, causing `ModuleNotFoundError`
- 🟡 Missing newline at end of `step_parser.py` (line 26)
- 🟡 Missing newline at end of `test_step_parser.py` (line 9)
- Test cannot be executed to verify passing status due to missing dependency

**Comment:**
The implementation is correct and follows the brief. However, the `pythonocc-core` dependency must be installed before tests can run. This is a setup/dependency management issue, not a code issue. Once the dependency is available, the tests should pass based on code analysis. Recommend adding a `requirements.txt` or `pyproject.toml` with the pythonocc-core dependency.