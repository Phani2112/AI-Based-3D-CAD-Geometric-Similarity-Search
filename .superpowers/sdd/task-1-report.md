### Task 1: STEP Parser Module - Report

- **STATUS**: DONE
- **Commits made**: eb0e74e, 149f090
- **Tests**: 1 passing / 1 total
- **Concerns**: None

**Summary**: Implemented STEPParser class in `src/data/step_parser.py` that parses STEP files using pythonocc-core. The implementation extracts faces from the parsed shape and exposes them via a `ParsedShape` wrapper class with a `.Faces` attribute. Test file created at `tests/test_step_parser.py` with a sample STEP file copied from the FabWave dataset. Tests pass successfully.
