# bayesian_panel_nmf — Documentation

Central reference for the Bayesian hierarchical panel model with low-rank
factorization. This document is built up alongside the codebase; sections are
added as features land.

> **Methodology note.** This is a low-rank factor-model approach: untreated
> outcomes are approximated by a small number of latent factors. It is **not**
> a difference-in-differences / parallel-trends design — the analysis relies on
> factor structure, not on matched pre-trends.

---

## Installation

Requires **Python 3.12–3.14** and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> bayesian_panel_nmf
cd bayesian_panel_nmf
uv sync --all-extras --dev
```

This installs the full stack, pinned in `pyproject.toml` and locked in
`uv.lock`:

| Package | Floor | Notes |
|---------|-------|-------|
| JAX / jaxlib | `>=0.10.2,<0.11` | jaxlib resolves transitively via jax — not pinned directly |
| NumPyro | `>=0.21,<0.22` | |
| ArviZ | `>=1.2,<1.3` | 1.x uses `xarray.DataTree` (not the legacy `InferenceData`) |
| NumPy | `>=2.0` | required by JAX 0.10 |
| pandas | `>=2.2.3` | |
| matplotlib / seaborn | `>=3.8` / `>=0.13.2` | hard dependencies — figures are a core feature |

### GPU / accelerator

JAX selects the accelerator wheel via its own extras (e.g. `jax[cuda13]`).
The default install is CPU-only; see the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
for CUDA/ROCm/TPU wheels. Chain parallelism is chosen automatically from the
visible devices (see `mcmc.auto_parallelism`).

### Verify the install

```bash
uv run python -c "import jax, numpyro, arviz; print(jax.__version__, numpyro.__version__, arviz.__version__)"
# expected: 0.10.x 0.21.x 1.2.x
```
