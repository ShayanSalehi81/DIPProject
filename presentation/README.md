# Micro-Q-Instruct presentation

This folder contains the reproducible 32-slide academic presentation for the completed Phase 1
and Phase 1.5 outputs.

## Files

- `Micro_Q_Instruct_Presentation.html`: editable 16:9 slide deck.
- `Micro_Q_Instruct_Presentation.pdf`: rendered 32-page submission.
- `Micro_Q_Instruct_Presentation_preview.png`: contact-sheet visual audit.
- `data_notes.json`: selected image records and the source-output audit trail.
- `build_deck.py`: collects and optimizes assets, then rebuilds the HTML.
- `render_deck.py`: renders the PDF and verifies the page count.

## Rebuild

Create an environment with Pillow, WeasyPrint, and PyMuPDF, then run:

```bash
python build_deck.py
python render_deck.py
```

The deck is grounded in `DataDegradationPipeline/README.md`,
`DataDegradationPipeline/outputs`, `RobustnessCheck/README.md`, and
`RobustnessCheck/outputs`. The ten triptychs are labeled illustrative; aggregate claims come from
the complete reports and tables.
