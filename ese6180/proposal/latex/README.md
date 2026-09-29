# ESE 6180 proposal — L4DC LaTeX source

Two-page project proposal (excluding references) in the L4DC (PMLR) format
required by the [course project description](../../resources/ESE6180-26Fall-Final-Project-Description.txt).
The draft summarizes the [v3 brief](../brief.md): quadratic certificate learning,
one controller update constrained by a verified decrease margin, and supporting
experiments. See the [v3 revision notes](../discussions/v3-revisin-notes.md) for
scope and checks; [v2 notes](../discussions/v2-revision-notes.md) are historical.

## Layout

| Path | Purpose |
| --- | --- |
| `proposal.tex` | The proposal source. |
| `l4dc2026.cls` | L4DC 2026 class, copied unmodified from the provided template. |
| `template/` | The unmodified upstream sample (`.tex`, `.bib`) kept for reference. |
| `references.bib` | Local bibliography for this standalone draft. |
| `figures/cartpole_recovery_regions.png` | Supporting preliminary panels, omitted from the two-page draft. |
| `figures/make_recovery_regions.py` | Regenerates the supporting panels from stored evaluation data. |
| `proposal.pdf` | Built output. |

The draft uses its local `references.bib`, including the Boffi and Zhang
certificate-learning papers. `proposal-overleaf.zip` contains `proposal.tex`,
`references.bib`, and `l4dc2026.cls`; upload it as a new Overleaf project and
select `proposal.tex` as the main document. The build does not depend on
repository artifacts, literature PDFs, or figures.

The [email draft](../discussions/email-prof-matni-v3.md) includes an Overleaf-link
placeholder for replacement after upload. No email has been sent.

## Build

```sh
make            # pdflatex -> bibtex -> pdflatex x2
make clean      # drop .aux/.bbl/.blg/.log/.out
make figure     # regenerate supporting panels (needs the repo .venv and artifacts/)
```

The supporting figure is rendered by `figures/make_recovery_regions.py`, which reads
`artifacts/evaluations/cartpole-failure-boundary-v1/final/analysis/episode-summary-v4/summary.json`
and reuses the helpers in `src/active_eval_gym/plotting.py`, so the panels stay
consistent with the `_v2` figures in `docs/findings.md`. The `artifacts/` tree is
git-ignored, so regeneration needs a local copy of that evaluation.

## LaTeX toolchain

`l4dc2026.cls` loads `jmlr.cls`, which pulls in a fair amount of TeX Live beyond
`texlive-latex-base`. If `kpsewhich jmlr.cls` returns nothing, install the needed
trees into `~/texmf` without root:

```sh
apt-get download texlive-publishers texlive-latex-recommended \
                 texlive-latex-extra texlive-science texlive-fonts-recommended
for d in *.deb; do dpkg-deb -x "$d" ext/; done
mkdir -p ~/texmf && cp -r ext/usr/share/texlive/texmf-dist/* ~/texmf/
```

`~/texmf` is on the default `TEXMFHOME` search path, so no further configuration
is needed.

## Deviations from the template worth knowing

- `\documentclass{l4dc2026}` is used without the `final` option. `final` adds the
  proceedings editor line, which is meaningless for a course proposal and costs
  three lines of a very tight two-page budget.
- The first-page proceedings banner ("Proceedings of Machine Learning Research vol
  vvv:1--N, 2026") is suppressed from `proposal.tex`, by appending a
  head-clearing step to the class's `jmlrtps` page style. Emptying `\jmlryear`
  would blank it too, but would also drop the `(C) 2026 S. Chen.` footer; the
  class file itself is left unmodified.
- The `keywords` block from the sample was dropped for the same reason. The
  course's required elements (title, team member, abstract, related work, problem
  formulation, goals) are all present; keywords are not among them. Re-add it if
  you would rather cut a sentence elsewhere.
- The proposal body occupies two pages; references start on a separate third
  page. The preliminary figure remains supporting material so the body can
  explain the control, evaluation, and improvement loops at the class's normal
  font size and margins.
- The foundation is a discrete-time quadratic specialization of certificate
  learning. The required main analysis connects verified decrease margins to
  one restricted gain update and examines cost regression. Expansion or a
  local nonlinear extension is optional; repeated PPO rounds are not required.
- A brief AI-assistance statement records drafting and preliminary-derivation
  assistance, following the syllabus guidance.
