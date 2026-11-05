# Short abstract options and M06 revision

The selected title is **Evaluation-Guided Domain Randomization: A Feedback Loop for
Controller and Certificate Learning**. It is applied in `midterm_report.tex`,
including the PDF metadata. The 75-word balanced abstract (option 1) is selected
and applied in the source.

## Shared report changes

M06 now opens with:

> Following Boffi et al. (2021), we bound the gap between sampled and population
> certificate loss for a frozen controller. Comparing each P with a nearby
> candidate P_epsilon in a finite set splits the gap into three differences.

The diagram comes immediately afterward. The net size, dimension, Lipschitz
constant, and concentration argument follow it. The displayed bound, its
applicability to a fitted certificate, the violation-risk inequality, and the
fresh-trajectory requirement remain. The temporary symbol M_K is expanded into
its definition. The comparison with Boffi et al.'s faster rates is removed.

The opening motivation is shortened by one rendered line to keep the performance
equation on page 1 with a six-line abstract. Font sizes, margins, figure sizes,
vertical-spacing commands, the diagram, and the displayed bound are unchanged.

## Validation

All three alternatives passed `ese6180/build_report.py --target midterm`: four
main-text pages, references beginning on page 5, no overfull boxes, and no undefined
citations or references. The selected balanced abstract is now the report abstract.
The proof paragraph and final-page layout were visually inspected.

The local PDF links below point to ignored build artifacts. Each build folder also
contains the exact candidate `midterm_report.tex`. The alternatives differ only in
the abstract; all use the same title, M06 rewrite, and opening motivation.

## 1. Balanced research style (75 words)

Our feedback loop uses policy evaluation to focus training near a performance threshold.
For known linear systems, learned and verified Lyapunov certificates constrain
controller updates to preserve stability on retained systems. We bound certificate
generalization error from finite-horizon trajectories and require Armijo decrease for
accepted steps. Preliminary experiments show that targeting gains depend on competing
controller changes across systems. Ongoing work seeks conditions for increasing the
fraction of acceptable conditions under a fixed evaluation distribution.

[PDF preview](../../Evaluation_Guided_Domain_Randomization/build/abstract-options/balanced/midterm_report.pdf)

## 2. Accessible, with the CartPole pilot (75 words)

We use evaluation to focus training on conditions close to acceptable performance. For
known linear systems, learned and verified certificates limit updates to preserve
stability on retained systems. We bound how accurately sampled trajectories estimate
expected certificate loss. Linear experiments show the strongest targeting gains when
systems favor competing controller changes. A CartPole pilot expands successful recovery
conditions while losing some previous successes. We seek conditions for net coverage
gains under a fixed evaluation distribution.

[PDF preview](../../Evaluation_Guided_Domain_Randomization/build/abstract-options/accessible/midterm_report.pdf)

## 3. Theoretical emphasis (70 words)

We connect certificate learning and domain-randomized control in an evaluation-guided
loop for known linear systems. Finite-horizon evaluations target training near a
performance threshold; verified Lyapunov certificates constrain controller updates. We
derive a uniform certificate generalization bound and a stability-preserving condition
for Armijo updates on retained systems. Preliminary experiments relate targeting gains
to competing controller changes across systems. Ongoing work seeks conditions for
increasing acceptable coverage under a fixed evaluation distribution.

[PDF preview](../../Evaluation_Guided_Domain_Randomization/build/abstract-options/theoretical/midterm_report.pdf)
