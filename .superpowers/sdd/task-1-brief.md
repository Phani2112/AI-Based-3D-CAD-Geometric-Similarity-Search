### Task 1: STEP Parser Module

**Files:**
- Create: `src/data/step_parser.py`
- Test: `tests/test_step_parser.py`

**Interfaces:**
- Consumes: STEP file path
- Produces: `TopoDS_Shape` object with faces and edges

**Step 1: Write the failing test**

```python
import pytest
from src.data.step_parser import STEPParser

def test_parse_step_file():
    parser = STEPParser()
    shape = parser.parse("tests/data/sample.stp")
    assert shape is not None
    assert hasattr(shape, 'Faces')
    assert len(shape.Faces) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_step_parser.py -v`
Expected: FAIL with "module not found"

**Step 3: Write minimal implementation**

```python
# src/data/step_parser.py
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

class STEPParser:
    def parse(self, filepath):
        reader = STEPControl_Reader()
        status = reader.ReadFile(filepath)
        if status == IFSelect_RetDone:
            reader.TransferRoots()
            shape = reader.OneShape()
            return shape
        return None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_step_parser.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/step_parser.py tests/test_step_parser.py
git commit -m "feat: add STEP parser using pythonocc-core"
```

Report file: `/home/jose-draeger/workspace/CADGCL V2/.superpowers/sdd/task-1-report.md`