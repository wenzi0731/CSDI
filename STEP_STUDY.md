# Direct CSDI vs DLinear + Residual CSDI

This experiment tests whether residualization reduces the performance penalty of
using a shared sampling budget across variables. It does not assume that the
hypothesis holds. The task is **history-conditioned multivariate forecasting**;
unconditional or metadata-conditioned scenario generation needs a different first
stage and is not implemented here.

## Run

From this repository root, using Python 3.10+:

```bash
python -m pip install -r requirements-step-study.txt
python -m unittest discover -s tests -v
python -m step_study.run --synthetic --smoke --seeds 42 \
  --output step_study_runs/smoke
```

The smoke run trains each model for one epoch and creates demonstration curves.
These are pipeline checks, **not evidence for the paper's claim**.

For real observations, sort rows chronologically, use regularly spaced complete
numeric observations, and explicitly select the variables (exclude timestamps,
identifiers and any future-unavailable features):

```bash
python -m step_study.run --data data/my_series.csv \
  --columns wind,solar,load --device cuda:0 --seeds 42 43 44 \
  --output step_study_runs/energy
```

Column names above are examples. Alternatively, supply a NumPy `.npy` array of
shape `[time, variables]`. Missing/nonfinite values are rejected; imputation and
resampling should be explicitly designed before using this entry point.

Edit `config/step_study.yaml` for history, horizon, stride, training epochs,
diffusion settings and sampling budgets. For memory constraints reduce evaluation
batch size or sample count; sample count must remain matched across both arms.
Forecast windows are batched with their ensembles, so attention memory grows with
batch size times ensemble size. Original CSDI scripts remain available.

Replay the saved models with exactly the original arguments plus
`--evaluate-only` (including `--smoke` for a smoke run). This replaces score/figure
files in that run directory without retraining or changing checkpoints. Fresh
training refuses to overwrite a nonempty directory. Evaluation-only checks the
resolved config, data hash, scaler, variable order and seed list. To change the
sampling grid, edit the initial configuration before training; this command
currently enforces exact configuration replay.

## Controlled comparison

| Arm | Diffusion training target | Final generated forecast |
| --- | --- | --- |
| direct | standardized future Y | generated Y |
| residual | Y - frozen DLinear(history) | DLinear(history) + generated residual |

Both arms use upstream `diff_CSDI`, the same time and variable attention, parameter
count, history-only conditioning, initialization, batch order, training noise,
diffusion schedule and optimization budget. All variables are jointly generated;
there is no feature subsampling. The baseline is not concatenated as an extra
input, so the denoiser's input shape and conditioning information are identical.

DLinear implements moving-average trend/seasonal decomposition with individual
linear layers by default. It is trained on training windows, selected using
validation MSE, frozen, and then used to define residual training targets.
It adds parameters and training cost to the complete system; only the diffusion
denoisers have equal capacity. Training residuals are in-sample; if first-stage
overfitting is material, add a cross-fitted first-stage experiment before making
mechanistic claims. No validation/test targets enter model optimization.

Raw time is split 60/20/20. Every target window stays within its own split. A
validation/test history may use earlier observations, as in rolling-origin
forecasting; previous test observations can become history at subsequent origins.
Scaling uses only training observations and is shared by both arms; residuals
are **not** separately standardized. Set stride at least horizon for nonoverlapping
targets if that suits the evaluation protocol.

Checkpoint selection uses fixed validation diffusion noise/timesteps for each
epoch and arm; the selected checkpoint is fixed over the entire step scan.
This validation objective is a Monte Carlo estimate, not an all-timestep average.
DDIM eta=0 uses a uniform index grid, traverses from the same terminal noise time
to clean time, and performs exactly K denoiser evaluations. It does not truncate a
longer chain at an intermediate noise level. Initial Gaussian samples are reused
across all K and both arms for each evaluation split/seed. Dropout is disabled.
The default 1000-step training schedule ends at near-zero signal-to-noise ratio.
If changing that schedule, verify terminal alpha is small enough for the Gaussian
sampling prior to be appropriate. No claim that DDIM is preferable to another
solver is made; use a matched second sampler to test robustness separately.

## Outputs

- `manifest.json`: resolved settings, train-only scaler, split boundaries, data
  hash, variable names, seeds, runtime versions, and repository base commit.
- `seed_*/{dlinear,direct,residual}.pt` and `*.history.json`: best checkpoints and
  epoch losses. `residual_diagnostics.json`: validation variance and baseline MSE.
- `per_variable.csv`: empirical ensemble CRPS in original units and divided by
  the training std, plus ensemble-mean RMSE in standardized units.
- `joint_scores.csv`: energy score over the whole standardized multivariate
  forecast trajectory and sampling time (residual includes DLinear inference).
- `performance_steps_{val,test}_*.{png,pdf}`: per-variable curves, paginated at 12
  variables; shaded bands are population SD across training seeds, not confidence
  intervals. A single seed has zero band width.
- `shared_step_regret.{png,pdf}` and `summary.json`: shared-step diagnostics.

For lower-is-better standardized CRPS M_i(K), the summary computes validation
K_i* = argmin_K M_i(K), population std(log K_i*), and mean relative regret
R(K) = mean_i [(M_i(K)-min_K M_i(K))/max(min_K M_i(K),1e-8)]. Ties choose the
smallest K. Also report each variable's near-optimal step set (default within 2%
of its minimum), and their common intersection. Broad flat curves can have noisy
argmins; do not interpret argmin dispersion alone as a generation-difficulty
measure.

A shared K is selected by mean standardized **validation** CRPS. Both that K and
the validation-selected per-variable K_i* are then evaluated on test. Their test
gap can be negative. Test-oracle regret over the grid is separately labeled
descriptive and must not be presented as held-out model selection. Per-variable
optima are diagnostics, not a proposed sampler mixing coordinates from different
joint trajectories. Energy score provides a joint-distribution check alongside
the marginal CRPS curves.

Flatter final-output curves alone do not prove residual difficulty became more
homogeneous: smaller residual amplitude and a dominant deterministic baseline can
also flatten absolute scores. Use relative regret, near-optimal overlaps, multiple
training seeds, adequate generation quality and joint scores together. Residual
variance reduction is a descriptive proxy, not a direct measure of diffusion
difficulty. Additional residual re-standardization and sampler ablations can
help distinguish amplitude effects from changed score-learning difficulty.

## Sources

- [CSDI, NeurIPS 2021](https://github.com/ermongroup/CSDI): denoiser reused from this repository.
- [DLinear / LTSF-Linear](https://github.com/cure-lab/LTSF-Linear): moving-average decomposition and linear forecasting design; implementation here is self-contained.
- [DDIM](https://arxiv.org/abs/2010.02502): deterministic resampled reverse process.
