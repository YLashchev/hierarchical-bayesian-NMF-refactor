# TODO — Over-Engineering Refactor Handoff & Model Diagnostics

## Refactor status (as of 2026-07-12)

### Complete (Tasks 1–10, all reviewed clean)

| Task | Commit | Phase gate |
| --- | --- | --- |
| T1 baseline capture | `47243a2` | 204 tests, 74% cov, 251s smoke |
| T2 convergence gate (TDD) | `a51b37f` | 3 new tests, ArviZ rank-normalized R-hat/ESS |
| T3 wire gate, delete hand-rolled diagnostics | `3614b07` | 194 tests, posterior PASS, cov 77%, wall +8.4% |
| T4 delete progress/heartbeat machinery | `4462ec1` | 188 tests |
| T5 delete parallel.py, inline worker resolution | `9f32b54` | 187 tests, posterior PASS, cov 76%, wall +2.1% |
| T6 minimal logging | `62a12ce` | 185 tests, logging_config 91→15 lines |
| T7 drop aliases, unexport validators, remove separate_texas | `beba165` | 184 tests |
| T8 fixed filenames, config consolidation | `ac437cb` | 185 tests, posterior PASS, cov 76%, wall −0.8% |
| T9 outcome-column helper (4 of 8 sites) | `55052e8` | 190 tests |
| T10 privatize PPC exports, NetCDF-only traces | `5529c07` | 190 tests, posterior PASS, cov 77%, wall +0.3% |

**Net: −1094 lines (329 insertions, 1422 deletions). Modules deleted: `parallel.py`, `scripts/check_diagnostics.py`. Tests: 190 (was 204). Coverage: 77% (was 74%). All four phase gates passed.**

### Task 11 (final verification) — BLOCKED, not a refactor defect

- Steps 1–3 (repo scrub, AGENTS.md update, line-count accounting): done.
- Step 5 (standard-run PASS-path validation, `fertility_config --type total`): **does not converge**.
  - Rank 5: R-hat 2.29, bulk ESS 2.5, 0 divergences, 49m34s.
  - Rank 7 (JAMA-supplement-selected): R-hat 2.17, bulk ESS 5.2, 0 divergences, 68m48s.
  - This is a **pre-existing model identification issue**, not a refactor regression.
  - Verified: `dev` and refactor branch produce **bit-identical posterior arrays** under the same rank-7 settings.
  - Verified: upstream `afranks86/dobbs_fertility` and our model have **identical data, prepared arrays, and log-joint**.
  - Verified: upstream **never computed R-hat/ESS/divergences** — no MCMC convergence diagnostics exist in their repository.
- Step 6 (final suite + ruff + mypy): all green (190 passed, ruff clean, mypy clean).
- Step 4 (delete baseline fixtures): **not yet done** — fixtures still present.
- Step 7 (Engram session summary): not yet done.

### Remaining cleanup steps

- [ ] Delete `tests/fixtures/pre_refactor_*` and `tests/fixtures/compare_posterior.py`.
- [ ] Decide on the uncommitted `configs/fertility_config.yaml` change (4 chains / 1000 warmup — diagnostic settings).
- [ ] Commit `AGENTS.md` update (uncommitted in main checkout).
- [ ] Commit `TODO.md` (untracked in main checkout).
- [ ] Write final Engram session summary.

---

## Model convergence diagnosis

### Root cause: state/time fixed-effect scale non-identifiability

The model defines:

```python
state_fe ~ ImproperUniform(R+)    # improper flat prior on positive reals
time_fe  ~ Gamma(1, 1)            # weakly informative
fixed_effects = log(state_fe) + log(time_fe)
```

For any constant `c`: `state_fe *= c` and `time_fe /= c` leaves the likelihood unchanged. This creates an **exact scale ridge** with no identifying constraint.

**Evidence from rank-7 trace**:

| Quantity | R-hat | Min bulk ESS |
| --- | ---: | ---: |
| Raw `state_fe` | 2.16 | 5 |
| Raw `time_fe` | 2.17 | 5 |
| Centered `state_fe` | 1.05 | 83 |
| Centered `time_fe` | 1.04 | 138 |
| `mu_ctrl` (the sum) | 1.05 | 93 |
| `mu` (with treatment) | 1.02 | 370 |
| `te` (treatment effect) | 1.07 | 48 |

- Correlation of mean log-state vs mean log-time: **−0.999994**.
- Per-chain regression slope: approximately **−1.000**.
- Chains drift along the ridge: state effects up +5.8 to +12.1 log units, time effects down by the same; their sum changes only ~0.001–0.004.
- The transformed log density grows ~3·delta along the invariant direction (D=51 states, N=48 time periods). This is consistent with an **improper posterior**.

### "Model feedback" — three compounding routes

1. **State ↔ time scale ridge** (dominant): state and time fixed effects trade a global offset with no anchor.
2. **Factor label switching**: NMF factors are identified only up to permutation. `unit_weight` R-hat 1.48, `time_fac` R-hat 1.32.
3. **Factor surface ↔ fixed effects**: row/column means of the factor surface can absorb variation belonging to state/time intercepts.

Removing the global scale (Route 1) drops R-hat from 2.16 to ~1.05, but Routes 2 and 3 remain.

### What does NOT fix it

- **Rank change** (5→7): improved aggregate R-hat 2.51→2.17, but did not eliminate the ridge.
- **Thinning change** (5→10): tested post-hoc on existing trace — no material improvement, some quantities get worse.
- **More chains** (2→4): confirmed failure is structural, not a 2-chain artifact.
- **More warmup** (1000 vs 2000): the ridge is an identification problem, not a warmup problem.

### Related GitHub issue

- [Issue #7](https://github.com/YLashchev/hierarchical-bayesian-NMF-refactor/issues/7): "Fix model fit failure with units having zero or near-zero counts due to improper uniform state fe prior."
  - Correctly identifies the improper `state_fe` prior as problematic.
  - But its zero-count trigger is absent from the total panel (minimum count 745; zero zero-count cells).
  - Should be broadened or linked to a separate issue for the global state/time non-identifiability.

---

## Upstream replication verification

### Data

- `data/raw/fertility_data.csv` is **byte-identical** to upstream `data/fertility_data.csv` (SHA-256 `c3b88a24...`).
- Prepared model arrays (`Y`, `denominators`, `control_idx_array`, `missing_idx_array`, state ordering, time ordering) are **cell-for-cell identical** when running upstream `clean_dataframe()` + `prep_data()` vs local `load_and_prepare()`.

### Model

- Local and upstream model have **identical sample sites**, identical `mu_ctrl`, `te`, `mu`, and **exactly equal total log joint** at a fixed rank-7 posterior draw (absolute difference 0.0).
- Local code adds validation guards, removes debug prints, and fixes Poisson/None-mask edge paths — none of which affect the NB training target.

### Inference

- `dev` and refactor branch have **byte-identical model files** and **identical NUTS/MCMC/RNG/Predictive calls**.
- The only meaningful addition is `extra_fields=("diverging",)` for divergence collection — cannot affect sampling.
- Apples-to-apples rank-7 run on `dev`: **every posterior variable is bit-for-bit identical** to the worktree branch.
- Draws CSV: **same SHA-256**.

### Upstream convergence check

- Grep over ALL upstream `.py`, `.qmd`, `.R`, `.rmd` files for `rhat`, `ess`, `divergences`, `convergence`, `print_summary`, `monitor`: **zero matches for MCMC diagnostics**.
- Upstream saves only `mu` (mu_ctrl), `te`, `disp`, `ypred` — derived quantities only. Raw `state_fe`, `time_fe`, `time_fac`, `unit_weight` are never saved or inspected.
- The supplement's §6.2 "Model Diagnostics and Criticism" are **PPC plots**, not MCMC convergence checks.
- Upstream's rank-selection criterion (PPC pass, 0.1 < P(T_pred > T_obs) < 0.9) verifies posterior predictive adequacy, not chain mixing.

### Supplement results comparison

Our rank-7 posterior draws match the published JAMA supplement tables within Monte Carlo noise:

| State | Metric | Supplement | Our Run | Diff |
| --- | --- | ---: | ---: | ---: |
| Texas | Excess | 16,161 | 15,869 | −292 |
| Texas | Pct change | 2.32% | 2.28% | −0.04 |
| West Virginia | Excess | 110 | 110 | 0 |
| West Virginia | Pct change | 0.80% | 0.80% | 0.00 |
| Wisconsin | Excess | 563 | 519 | −44 |
| Wisconsin | Pct change | 0.96% | 0.88% | −0.08 |

Differences are consistent with Monte Carlo variation from different retained draw counts (supplement: 250 draws/chain × 4; ours: 400 × 4).

---

## Reporting formula difference (reporting.py vs upstream R)

Our `reporting.py` and the upstream R `make_state_table()` use **different excess estimands**.

### Our formula (reporting.py)

```python
expected = sum(exp(mu))           # counterfactual: posterior predictive of mu_ctrl
excess   = observed - expected     # observed data minus counterfactual
pct      = excess / expected * 100
```

### Upstream R formula (fertility_paper_figures.qmd, lines 390–460)

```r
treated   = sum(exp(mu_treated))   # model's fitted value WITH treatment
untreated = sum(exp(mu))           # counterfactual WITHOUT treatment
excess    = treated - untreated     # posterior treatment effect
pct       = 100 * (treated_rate / untreated_rate - 1)
```

### Impact on published numbers

| State | Our formula | Upstream formula | Supplement |
| --- | ---: | ---: | ---: |
| Texas excess | 17,229 | 15,869 | 16,161 |
| Texas pct | 2.47% | 2.28% | 2.32% |
| West Virginia excess | 53 | 110 | 110 |
| Wisconsin excess | 482 | 519 | 563 |

- Our formula weights actual observed data; the upstream formula uses the model's posterior fitted values.
- Neither is wrong — they are scientifically different estimands.
- **To reproduce the supplement's published values, use the upstream formula.**

### Action item — DONE 2026-07-12

- [x] Adopted the upstream formula: `_compute_per_unit_post_treatment` now reports `excess = sum(exp(mu_treated)) − sum(exp(mu))` and `excess_pct = 100 * (treated/untreated − 1)`, matching the supplement's per-state "Expected difference" and "Expected percent change". Public CSV columns preserved; only `excess_*` semantics shifted from observed-minus-counterfactual to model-implied treatment effect. `observed` kept informational.
- [x] Confirmed PPC formulas (`make_abs_ppc_plot`, `make_acf_ppc_plot`, `make_rmse_ppc_plot`, `make_unit_corr_ppc_plot`) already match upstream R (`outcome − exp(mu)` and `ypred − exp(mu)` on `treatment==0` rows). No PPC code change.
- [x] Confirmed `make_summary_table` (Table 1) and `make_interval_plot` (method=mu) already used `treated − untreated`.
- [x] Test `test_per_unit_post_treatment_uses_mu_not_ypred` updated to assert the new estimand (`excess_mean = 3*100*(e^0.1−1) = 31.55`, `excess_pct_mean = 100*(e^0.1−1) = 10.52`).
- [x] Verification: 190 tests pass, ruff clean, mypy clean, real draws CSV `results/total/NB_births_total_3.csv` regenerated end-to-end successfully.
- Methodological context: this is a reported-estimand change (the supplement-equivalence fix the user approved), not a refactor. The previously reported `observed − untreated` conflated observed-vs-fit residual with excess; the supplement publishes the model treatment effect (`treated − untreated`). For Texas rank 3, values shifted from 25,835 → 24,533 exactly by the observation-vs-fit gap.

---

## Methodological assessment (researcher subagent)

**Is it acceptable to use an unconverged posterior?** **No.**

- R-hat > 1.01 (let alone 2.16) means draws are not representative of one common posterior.
- Convergence is **global** (Stan manual): a stable derived estimand (R-hat ~1.04) does not validate raw parameters at 2.16.
- The only exception is a known symmetry with invariant quantities — but even our `expected_total` R-hat was 1.04 (above 1.01) and `te` was 1.07.
- An improper posterior under NUTS produces transient non-stationary draws that look plausible but are mathematically invalid.
- **Recommendation**: Do not publish or substantively interpret this fit. Remove the invariance (sum-to-zero or reference-level constraint), impose proper priors on deviations, then rerun dispersed chains with full diagnostics.

Sources: Vehtari et al. (2021), Stan Reference Manual ("Convergence is global"), Stan User Guide ("Problematic Posteriors").

---

## Poisson vs Negative Binomial: why different for fertility and mortality

### Fertility (high counts → NB with κ=1e-4)

Births per state-bimonthly cell have **mean ~12,000** and empirical **variance-to-mean ratio ~121** (massively overdispersed). Poisson assumes variance = mean, so it would understate uncertainty by a factor of 120×. The supplement explicitly:

- Adapts Bayesian NMF literature (Cemgil 2009; Schein 2016) to build a Poisson-log-normal → NB marginal.
- Marginalizing multiplicative gamma noise ν over Poisson gives NB with mean µ and dispersion κ: `Var(Y) = µ(1+κµ)`.
- Chooses fixed `κ=1e-4` after finding "standard hyperprior distributions for κ yielded poor mixing" (supplement line 546).

### Mortality (rare events → Poisson)

Infant deaths are rare (low counts per cell). At low means, overdispersion's quadratic term `κµ²` is negligible. Poisson is both physically natural (independent rare events) and computationally simpler (avoids an unidentifiable dispersion parameter).

---

## Known issues and minor notes from reviews

1. **analysis_workers silent coercion** (Task 5): `analysis_workers: 0` or `<-1` now silently coerces to 1 instead of raising `ValueError`. Spec-intent but the behavior change was not disclosed in the commit message. Consider adding a warning or validation if safety matters.

2. **_format_elapsed retention** (Task 4): The brief listed `_format_elapsed` for deletion, but it's still called by the per-type "complete in Xm Ys" timing line. Accepted deviation (brief was internally inconsistent).

3. **Visualization sites 5-8 deferred** (Task 9): The `_detect_outcome_column` helper was routed through only 4 of 8 call sites. Sites 5-8 (`make_unit_fit_plot`, `make_unit_gap_plot`, `make_interval_plot`, `make_summary_table`) never raise on absence and don't check `"outcome"` explicitly — routing them through the always-raising helper would be a behavior change. Deferred as future work.

4. **README heartbeat scope creep** (Task 8): Two stale README sentences about the Task-4-deleted heartbeat were updated. Accurate, harmless, self-flagged, kept.

5. **AGENTS.md** (main checkout): Updated but uncommitted. Changes: removed check_diagnostics.py, --save-diagnostics, parallel.py; added always-on convergence JSON; fixed stale comment.
