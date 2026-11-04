# Research note and midterm hand-in

- `main_report.tex` is the detailed research note, including full foundation
  proofs and proposed update results.
- `midterm_report.tex` is the separate L4DC hand-in: four pages of main text,
  followed by references. It explains the loss hierarchy, the three-arrow
  uniform-bound argument, the guarded update, and preliminary evidence.
- Both documents share `references.bib`. Only cited entries are printed.

From the repository root:

```bash
python3 ese6180/build_report.py --target midterm
python3 ese6180/build_report.py --target research
```

The default target remains `research`. PDFs and logs go to this directory's
ignored `build/` folder. Use `--output-dir PATH` for a separate build directory.
The midterm build checks the last main-text page and fails if it exceeds four;
references start after an explicit page break. Standard L4DC margins and body
font size are retained.

## Figures and evidence

The committed `midterm_assets/linear_results.pdf` permits compilation without
the local experiment artifacts. Its PNG is a preview. The accompanying
`figure_provenance.json` identifies the retained inputs, SHA-256 hashes, sample
sizes, probability budget, replicate counts, and extracted results.

To regenerate from the retained local artifacts:

```bash
uv run python scripts/make_midterm_figures.py
```

This command analyzes saved data and collects no new experiments. The left
panel uses 500 certificate datasets per sample size from the declared stress
fixture; its failure budget is 0.01 per size (0.05 jointly across five sizes).
The right panel uses both predeclared smoke-study replicates, with their
individual observations shown. It does not represent the unrun full preset.

The notebook and long note are learning/research material; they are not required
to read the hand-in. The hand-in distinguishes written proofs, the short
preservation derivation, implemented mechanisms, and remaining proof tasks.
It claims neither universal curriculum improvement nor a CartPole/PPO theorem.

## Source attribution

- Boffi et al. (2021, proceedings of CoRL 2020): trajectory-margin certificate
  learning. The normalized discrete-time loss and elementary bound are project
  specializations, not reproductions of the paper's strongest rates.
- Armijo (1966): sufficient-decrease backtracking. The certificate guard is a
  project addition.
- Florensa et al. (2017): performance-adaptive starting-state curricula.
- Berkenkamp et al. (2017): state-space safe-region expansion under additional
  model assumptions, discussed as context for a possible extension.

The scalar witness uses its own initial step 0.4 and Armijo coefficient 0.5.
It is an asymmetric boundary target; it does not demonstrate failure of every
symmetric boundary rule. Loss quantiles in the figure are not confidence
intervals, and reference-grid coverage is not continuous-domain area.
