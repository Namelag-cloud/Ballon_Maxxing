# Candidate crop dataset

Put labeled candidate crops in the train/val/test class directories.

Recommended first classes:

- `balloon`: genuine normal balloon cores.
- `ui`: score digits and other colored UI glyphs that the detector mistakes for balloons.
- `notification`: white notification balloons.
- `unknown`: ambiguous/uncertain crops that should not be confidently clicked.

Start with a roughly balanced dataset. Keep a held-out set of UI glyphs from different frames so validation is not just memorization of one screenshot.
