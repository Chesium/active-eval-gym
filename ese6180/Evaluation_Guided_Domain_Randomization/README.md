# Research note and midterm hand-in

- `main_report.tex` is the detailed research note, including full foundation
  proofs and proposed update results.
- `midterm_report.tex` is the separate L4DC hand-in: four pages of main text,
  followed by references. It explains the feedback loop, the loss hierarchy, the
  uniform bound, the guarded update, preliminary evidence, and open challenges.
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

The committed `midterm_assets/linear_results.pdf` and `ppo_loop.pdf` permit compilation without the local experiment artifacts. The PNG is a preview.
The accompanying `figure_provenance.json` identifies the retained inputs, SHA-256
hashes, sample sizes, probability budget, replicate counts, and extracted results.
`update_diagnostics.json` holds the step-size diagnostics quoted in the report.

To regenerate from the retained local artifacts:

```bash
uv run python scripts/make_midterm_figures.py
uv run python scripts/midterm_update_diagnostics.py
```

Panels (b) and (c) come from a crossed study, described in
`docs/linear-crossed-study.md`. It builds on the family study in
`docs/linear-family-study.md`, which must run first:

```bash
uv run python scripts/linear_family_study.py run --preset full --jobs 16 \
  --output artifacts/linear-family-study-v1
uv run python scripts/linear_crossed_study.py run --preset full --jobs 16 \
  --output artifacts/linear-crossed-study-v1
uv run python scripts/linear_crossed_study.py analyze \
  --run artifacts/linear-crossed-study-v1 \
  --family-run artifacts/linear-family-study-v1 \
  --output artifacts/linear-crossed-study-v1-report
```

Figure 2 comes from `midterm_assets/ppo_loop_data.json`, a compact record of the
development PPO study `cartpole-ppo-boundary-v5-adaptive`, which lives on another
machine. `scripts/extract_ppo_loop_data.py` produced it from that study's saved files.
The record keeps all ten rounds; the figure plots through 153,600 transitions. Blue
sets are held-out reference evaluations, and red frontiers are the discovery results
that chose each round. The study has no matched control and is not confirmatory.

The linear runs write about 2.9 GB. In WSL, check free space on the Windows drive first.

The first command analyzes saved data and collects no new experiments. The second
reruns deterministic linear rollouts. It stops unless every recorded round is
reproduced, and only then varies the backtracking factor. The left panel uses 500
certificate datasets per sample size from the declared stress fixture; its failure
budget is 0.01 per size (0.05 jointly across five sizes). The other panels show means
over every declared family with a shared certificate. Step-grid gaps from the original
smoke study are artifacts, so the report defines its update with the exact step.

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

The scalar witness uses its own initial step 0.4 and Armijo coefficient 0.5.
It is an asymmetric boundary target; it does not demonstrate failure of every
symmetric boundary rule. Loss quantiles in the figure are not confidence
intervals, and reference-grid coverage is not continuous-domain area.
