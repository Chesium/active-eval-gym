Project proposal requirement:

- 2 pages maximum (not including references)
- title
- team members
- abstract
- related works
- problem formulation
- goals
- clearly identify:
  - the feedback loop in your problem formulation
  - what is uniquely L4DC about your proposed work.


---

Idea:

- [Trunk] Level set estimation to adaptively evaluate policy performance
  - [Theory] specialize the general cases to linear systems / some control scenario and derive some bond/sensitivity for some other assumptions (see discussions)
- [Stretch] Iterative data distributions
  - Intuition: the points on the edge of the level set are critical for determining the failure boundary and also crucial for improving the model performance if added to the policy training data
  - adding sample from near the level set to the dataset may improve the policy
  - $D_0\to D_0 + \Delta\Longrightarrow \pi_0\to \pi_0+\Delta_\pi\Longrightarrow \text{LS}_0\to\text{LS}_0+\Delta_\text{LS}\Longrightarrow\cdots$
  - maybe we can somehow prove, in some sense, the "Volume" of the LevelSet will get larger and larger
  - relevant: Catastrophic Forgetting
  - relevant: barrier function

