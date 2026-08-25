# Label scope: what counts as positive and negative

This file is the authoritative definition of the gatekeeper's labels. Wherever a training or
evaluation report refers to the MVP label definition, it means this.

## 1. Positive class (record = 1)

A useful text screen, in one of the launch trigger scenes:

- Projector screen
- Text on a computer screen
- Slides
- Whiteboard
- Document page
- Code screen

## 2. Explicitly excluded

These do not count as a positive trigger, and they are also kept out of the training negatives
as ambiguous hard negatives:

- Text in a phone app interface
- Television and streaming menu text
- Product packaging and label text

All three are "has text but is not a launch scene", which is semantically fuzzy. Including them
blurs the binary decision boundary. The excluded subclasses are archived in
`data/processed/manifest_out_of_scope.csv`; the images are not deleted, they are only kept out
of training and evaluation.

## 3. Valid negatives

Clear cases of "should not be recorded": signage and street signs, book spines, phone lock
screens, screens playing video, and textless landscapes, portraits, interiors and food.

## 4. Why the scope was narrowed

2026-06-17, phase 3.4 to 3.4-B.

After a data expansion introduced the ambiguous hard negatives above:

| Metric | Wide boundary (with ambiguous negatives) | Narrowed |
|---|---|---|
| Aggregate test F1 | 0.70 | 0.756 |
| 5-seed variance | Did not narrow | Did not narrow |
| Positive FN on the clean original distribution | 0.206 | 0.135 |

Adding the ambiguous negatives lowered aggregate F1 *and* bought no reduction in variance, while
the model actually improved on the clean distribution. The diagnosis was label ambiguity rather
than insufficient data, so the scope was narrowed back to the launch definition in section 1.

The full sequence is in `docs/gatekeeper-training-log.md`.

Measurement note: numbers taken under the narrowed definition are not directly comparable with
numbers taken under the wide boundary. Every cross-definition comparison in the reports is
marked as such in its own table.
