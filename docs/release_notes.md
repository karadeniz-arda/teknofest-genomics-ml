# Release notes

## Public research release — 0.1.0

- Curated source code, synthetic schema example, tests, and methodology documentation are included.
- Competition datasets, organizer test files, submission outputs, trained model artifacts, identifiers, and historical experiment outputs are excluded.
- The package uses a standard `src/teknofest_genomics` layout and supports `pip install -e .` for scripts and tests.
- Documentation distinguishes final training (all labeled rows with balanced sample weighting) from the validation-only 80/20 OOF jury simulation.
- The documented deployment decision threshold is fixed at 0.50.
