# Adaptive frontier pilot

STATUS: COMPLETE. Main workflow exec9452 exited0. All training, evaluation,
analysis, figures, integrity checks, and reporting are complete. No background
jobs or subagents remain for this task. No additional experiment is authorized
by completion alone. Latest live usage at2026-11-04 22:41:35UTC:18%five-hour,
87%weekly,statusok. Fresh check required before future sustained work.

## User decisions and scope

User explicitly requested implementing evaluate → refresh boundary → retrain,
with critical design questions. Chosen: original nominal parent, all3existing
seeds0/1/2,10rounds×25600transitions=256000perseed,adaptiveonly with historical
fixed-frontier comparison, unchanged PPO/reward/duration correction, stop a seed
if discovery acceptance becomes empty and report remaining budget as unrun.
No uniform fallback or stale-frontier fallback. Also stop distinctly if all
conditions become accepted (no frontier left). Allseeds completedfullbudget;
no stop condition occurred, zero unrun transitions.

## Implementation

- Config `configs/experiments/cartpole_ppo_boundary_v5_adaptive.toml` references
  scientific settings fromv4-long and its verified initial stochastic calibration.
  Historical dense comparison usesv4-focus, an exact replay ofv4-long.
- `src/active_eval_gym/ppo_adaptive_curriculum.py`: separate explicit multiround
  runner; old single-round protocol remains guarded. Eachseed keeps live model,
  optimizer, environment, RNG, condition/resetstreams, anddurationEMA/counts.
  Discovery uses32episodes×143cells with freshrole1outer-round nuisance seeds.
  Selectunaccepted4-neighbours ofcurrentacceptedset,50%background/50%frontier.
  Targets stage at25600boundaries, applied atnext scheduled2560lawrefresh.
  Ongoing episodesretainconditions. Spawned discovery checks preserveRNG/weights.
  Checkpoints include sealedmodel, rawprefix, targetprovenance, optimizerhashes,
  durationstate. Interrupted jobs replayfromparent andverifycompletedrounds.
- `InverseLengthSampler.set_target_shares` stages newtargets without resetting
  currentlaw,EMA,counts. Newtests coveratomicvalidation andstatepreservation.
- `scripts/run_ppo_adaptive_curriculum.py` runs allseedtraining thenwrites an
  immutabletrainingbarrier before anyreferenceevaluation. Reference273cells,
  64pairedrole4seeds, everycompletedcheckpoint. No referencetargeting.
- `scripts/analyze_ppo_adaptive_curriculum.py` readsartifacts only, plots each
  seed's referencecoverage/changingdiscoveryoutline/desiredandactualexposure,
  andadaptive-vs-historicalcurves. No bestcheckpointselection.

## Outputs and results

Study `artifacts/experiments/cartpole-ppo-boundary-v5-adaptive`.
Workflow launched~21:55UTC, finished~22:39UTC on4November2026. Elapsed2630.1s
(43.8min), within35–60min estimate;9.5GiB. All12467evaluationjobs/669760episodes
complete (4004discoveryjobs128128episodes;8463referencejobs541632episodes).
Training768000additionaltransitions total. No collectionfailures/retries.

Final accepted referencecounts61/68/58 versusfrozenhistorical0/0/0. However,
allthree loseall28parentacceptedcells. Originalregionrecovery0/1792episodes
perseed;nominal0/64. Alloriginalregionlostby230400. Finalacceptance mostly
half-lengths.04,.05,.075,.10;seed1alsofour.20cells;none>=.30. This isdrift,
notoutwardexpansion, despitenonzerocoverage. Seed0nominalfailsby204800,
seeds1/2by230400. Earliergains/losses arefullytabulatedinreport.

Dynamicexposuretrackingalsofails: meanfinalroundshort<=.3target71.74%,actual
89.62%;original15discoveryacceptedcells target9.78%,actual.86%. Meaninterval
celltarget/exposureTV.255round1→.529round10. EMAstalenessduringpolicychange
isplausible, causalityuntested;no retentioninterventionwasrun. Empty-setstop
doesnotpreservetheparenttaskwhennewregionsremainaccepted.

Finalfigures underfigures/pilot/,allPNG+PDF:
`coverage-shift-seed-000`, `coverage-shift-seed-001`,
`coverage-shift-seed-002`, `adaptive-vs-frozen-learning-curves`.
AllfourPNGswerevisuallyinspected. Blackorangeoutlines nowchangewithdiscovery;
grayblueoutlineisparentreference. Lengthsare rankspaced (reportexplainsareas).

## Verification and reports

390testspassed68.21s beforelaunch,lintandwhitespacepassed. No numericalsource
changedafterlaunch. All30seals/rawcontentandfilehashes/lawschedules/frontier
reconstructions/sourcearchive/parent/calibrationverify. First25600reproducev4
exactly(all28oldrawcolumns,weights,optimizer,controller);parentandfirst-round
reference1092sharedcellsalsoreproduceexactly,includingtrajectoryaggregates.

Evidence `analysis/first-round-verification.json`, `training-integrity.json`,
`integrity-checks.json`, `summary.json`, `exposure-and-retention.json`.
Postrunartifact-onlyscripts `verify_adaptive_results.py`, `analyze_exposure.py`.
Launchsource/protocolarchived underprotocol/. Completion.jsonanddurablelog
logs/workflow.log. No needtoreruncompletedtraining/evaluations.

Fullprotocol/results/figurelinks:
`ese6180/discussions/2026-11-04-cartpole-ppo-adaptive-frontier-pilot.md`.
Mainplan Section13.10 updated in
`ese6180/discussions/2026-11-04-post-midterm-cartpole-ppo-plan.md`.
Reportfigurelinksresolve;gitdiff--checkpassed.

Allchangesremainuncommitted. No commit,push,cloudoperation,orconfirmatorylock.
Preserve unrelateduser`ese6180/discussions/a.md`andpriorchanges. Existing
v3/v4/v4-long/v4-focusdataremainunchanged. No furtherrequiredworkremainson
thisrequest. A future retention/duration-trackingexperimentneedsitsowndeclared
comparison;do notsilentlystartone.
