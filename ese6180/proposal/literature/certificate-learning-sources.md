# Certificate-learning papers for the v3 working proposal

Downloaded from the publishers' public PDF links on 2026-09-28. These are
candidate readings selected during proposal revision, not papers confirmed as
Prof. Matni's intended office-hour recommendations. Existing literature files
were preserved. Both files were checked for a PDF signature and parsed for
page count and title.

## Preferred foundation

**Nicholas Boffi, Stephen Tu, Nikolai Matni, Jean-Jacques Slotine, and Vikas
Sindhwani. Learning Stability Certificates from Data.** Proceedings of the
2020 Conference on Robot Learning, PMLR 155:1341–1350, published 2021.

- [Publisher record](https://proceedings.mlr.press/v155/boffi21a.html)
- [Download source](https://proceedings.mlr.press/v155/boffi21a/boffi21a.pdf)
- [Local PDF](boffi21a-learning-stability-certificates.pdf): 10 pages, 1,722,259 bytes
- SHA-256: `9eca38d878839b27352234486379c27a2f66dbcedf477b189202c0e40c4fdbfd`
- BibTeX key: `boffi2021learning` in [references.bib](../latex/references.bib)

Read Sections 3–4 for the certificate-learning formulation and generalization
analysis. The principal theory is continuous time and already includes quadratic
classes. Our proposed discrete-time finite-cover argument is a simpler adaptation;
it does not inherit the paper's rates or global conclusions without their
assumptions. Section 5's stronger conclusions are not a minimum reproduction
target. The CoRL conference year and PMLR publication year differ intentionally.

## Optional robustness context

**Thomas Zhang, Stephen Tu, Nicholas Boffi, Jean-Jacques Slotine, and Nikolai
Matni. Adversarially Robust Stability Certificates can be Sample-Efficient.**
Proceedings of the 4th Annual Learning for Dynamics and Control Conference,
PMLR 168:532–545, 2022.

- [Publisher record](https://proceedings.mlr.press/v168/zhang22a.html)
- [Download source](https://proceedings.mlr.press/v168/zhang22a/zhang22a.pdf)
- [Local PDF](zhang22a-adversarially-robust-stability-certificates.pdf): 14 pages, 394,830 bytes
- SHA-256: `98e8549f7b7d5fc134cf1029460e79cf1ba32334f76a20cc8157801158462304`
- BibTeX key: `zhang2022adversarially` in [references.bib](../latex/references.bib)

This connects adversarial perturbations, incremental stability, and statistical
complexity. It is context for a possible robustness extension, not a second
required theorem reproduction. The elementary gain-update inequality in our
notes is derived directly and is not attributed to this paper as its result.

## Existing complementary sources

- [Berkenkamp et al.](1705.08551v3.pdf): model-based learning and Lyapunov-based
  safe-region expansion; not a guarantee for unconstrained PPO curricula.
- [Control Barrier Functions: Theory and Applications](Control_Barrier_Functions_Theory_and_Applications.pdf):
  safety/invariance context, distinct from convergence and finite-horizon recovery.
- [Howard et al.](1810.08240v9.pdf): time-uniform inference for supporting rollout
  evaluation, distinct from generalization of a fitted Lyapunov candidate.
