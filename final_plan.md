# Final Plan — Paper Submission (Due: April 3, 2026)

## Current State
- **Code**: Gradient-based adaptive CP is implemented and tested (200-image subset: 14.9% median reduction, 90.05% coverage)
- **NPZ files**: Contain OLD MC Dropout-based adaptive results (only 1.4% improvement) — **must re-run**
- **Paper**: Section 3.4 still describes MC Dropout as the modulator — **must rewrite to gradient magnitude**
- **Paper**: XX placeholders throughout — **must fill after evaluation**

## Steps

### 1. Re-run full evaluation (~4 min)
```bash
uv run python -m scripts.run_evaluate
```
- Pipeline is already updated to use `gradient_sigma()` in step 5/6
- MC Dropout results loaded from existing npz (no recomputation)
- Outputs: `outputs/results_4x.npz`, `outputs/results_8x.npz` with gradient-based adaptive numbers

### 2. Regenerate all figures (~30 sec)
```bash
uv run python -m scripts.generate_figures
```
- Adaptive vs uniform interval figures now use gradient-based widths
- Three-way calibration curves, width histograms, coverage histograms all updated

### 3. Fix paper.tex — narrative + numbers
Changes needed:
- **Abstract**: Replace "MC Dropout variance as a spatial difficulty modulator" → gradient magnitude; fill XX%
- **Contributions** (item 1): MC Dropout → gradient magnitude
- **Related work** (last para of CP section): Update "normalises by MC Dropout standard deviation" → gradient magnitude
- **Section 3.4** (Adaptive CP): Rewrite entirely — replace MC Dropout std equations with gradient magnitude σ = |∇ŷ|^0.3
- **Table 1 caption**: "MC Dropout std" → "gradient magnitude"
- **Table 1 data rows**: Fill XX placeholders with real numbers from npz
- **Section 4.3** (Adaptive CP Results): Fill all XX placeholders
- **Discussion "MC Dropout as modulator" paragraph**: Rewrite — gradient magnitude is the modulator now, not MC Dropout
- **Conclusion**: Fill XX%

### 4. Compile paper
```bash
cd paper && pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

### 5. Git commit
Stage and commit all changes with descriptive message.

## Expected Results (from 200-image subset)
| Metric | 4x (expected) |
|--------|--------------|
| Adaptive coverage | ~90% |
| Median width reduction | ~15% |
| Smooth region reduction | ~55% |
| % pixels tighter | ~64% |
