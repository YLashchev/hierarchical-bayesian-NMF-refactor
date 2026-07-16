"""Unit tests for choose_mcmc_parallelism's device-aware chain_method
selection. These test the DECISION logic only -- no real MCMC, no real
JAX devices beyond whatever this test process's CPU backend provides
(only the platform-detection call is monkeypatched, never real sampling)."""

from unittest.mock import MagicMock

from bayesian_panel_nmf.parallelism import choose_mcmc_parallelism


def _fake_devices(platform: str, n: int) -> list:
    devices = []
    for _ in range(n):
        d = MagicMock()
        d.platform = platform
        devices.append(d)
    return devices


def test_single_cpu_device_uses_sequential_with_max_chains(monkeypatch):
    monkeypatch.setattr(
        "bayesian_panel_nmf.parallelism.jax.devices",
        lambda: _fake_devices("cpu", 1),
    )
    monkeypatch.setattr(
        "bayesian_panel_nmf.parallelism.jax.local_device_count", lambda: 1
    )

    num_chains, chain_method = choose_mcmc_parallelism(max_chains=4)

    assert num_chains == 4
    assert chain_method == "sequential"


def test_multi_cpu_device_uses_parallel_capped_at_device_count(monkeypatch):
    monkeypatch.setattr(
        "bayesian_panel_nmf.parallelism.jax.devices",
        lambda: _fake_devices("cpu", 4),
    )
    monkeypatch.setattr(
        "bayesian_panel_nmf.parallelism.jax.local_device_count", lambda: 4
    )

    num_chains, chain_method = choose_mcmc_parallelism(max_chains=4)

    assert num_chains == 4
    assert chain_method == "parallel"


def test_multi_cpu_device_caps_num_chains_below_max_chains(monkeypatch):
    """When n_devices < max_chains on a multi-CPU host, num_chains is
    capped to n_devices, not max_chains."""
    monkeypatch.setattr(
        "bayesian_panel_nmf.parallelism.jax.devices",
        lambda: _fake_devices("cpu", 2),
    )
    monkeypatch.setattr(
        "bayesian_panel_nmf.parallelism.jax.local_device_count", lambda: 2
    )

    num_chains, chain_method = choose_mcmc_parallelism(max_chains=4)

    assert num_chains == 2
    assert chain_method == "parallel"


def test_single_gpu_device_uses_vectorized_with_max_chains(monkeypatch):
    monkeypatch.setattr(
        "bayesian_panel_nmf.parallelism.jax.devices",
        lambda: _fake_devices("gpu", 1),
    )
    monkeypatch.setattr(
        "bayesian_panel_nmf.parallelism.jax.local_device_count", lambda: 1
    )

    num_chains, chain_method = choose_mcmc_parallelism(max_chains=4)

    assert num_chains == 4
    assert chain_method == "vectorized"


def test_multi_gpu_device_uses_parallel_capped_at_device_count(monkeypatch):
    monkeypatch.setattr(
        "bayesian_panel_nmf.parallelism.jax.devices",
        lambda: _fake_devices("gpu", 4),
    )
    monkeypatch.setattr(
        "bayesian_panel_nmf.parallelism.jax.local_device_count", lambda: 4
    )

    num_chains, chain_method = choose_mcmc_parallelism(max_chains=4)

    assert num_chains == 4
    assert chain_method == "parallel"


def test_single_tpu_device_uses_vectorized_with_max_chains(monkeypatch):
    """TPU single-chip mirrors single-GPU: vmap on one chip, not pmap.
    A literal ``parallel`` here would pmap across 1 device and fall back to
    sequential (and only 1 chain if num_chains were capped to n_devices)."""
    monkeypatch.setattr(
        "bayesian_panel_nmf.parallelism.jax.devices",
        lambda: _fake_devices("tpu", 1),
    )
    monkeypatch.setattr(
        "bayesian_panel_nmf.parallelism.jax.local_device_count", lambda: 1
    )

    num_chains, chain_method = choose_mcmc_parallelism(max_chains=4)

    assert num_chains == 4
    assert chain_method == "vectorized"


def test_single_tpu_device_caps_num_chains_below_max_chains(monkeypatch):
    """max_chains=2 on a single TPU device still yields 2 vectorized chains."""
    monkeypatch.setattr(
        "bayesian_panel_nmf.parallelism.jax.devices",
        lambda: _fake_devices("tpu", 1),
    )
    monkeypatch.setattr(
        "bayesian_panel_nmf.parallelism.jax.local_device_count", lambda: 1
    )

    num_chains, chain_method = choose_mcmc_parallelism(max_chains=2)

    assert num_chains == 2
    assert chain_method == "vectorized"


def test_multi_tpu_device_uses_parallel_capped_at_device_count(monkeypatch):
    monkeypatch.setattr(
        "bayesian_panel_nmf.parallelism.jax.devices",
        lambda: _fake_devices("tpu", 4),
    )
    monkeypatch.setattr(
        "bayesian_panel_nmf.parallelism.jax.local_device_count", lambda: 4
    )

    num_chains, chain_method = choose_mcmc_parallelism(max_chains=4)

    assert num_chains == 4
    assert chain_method == "parallel"


def test_unrecognized_platform_falls_back_to_sequential(monkeypatch):
    monkeypatch.setattr(
        "bayesian_panel_nmf.parallelism.jax.devices",
        lambda: _fake_devices("unknown_future_backend", 1),
    )
    monkeypatch.setattr(
        "bayesian_panel_nmf.parallelism.jax.local_device_count", lambda: 1
    )

    num_chains, chain_method = choose_mcmc_parallelism(max_chains=4)

    assert num_chains == 4
    assert chain_method == "sequential"


def test_max_chains_is_respected_as_upper_bound_on_single_cpu(monkeypatch):
    """max_chains=2 on a single CPU device still yields 2 chains, not 4."""
    monkeypatch.setattr(
        "bayesian_panel_nmf.parallelism.jax.devices",
        lambda: _fake_devices("cpu", 1),
    )
    monkeypatch.setattr(
        "bayesian_panel_nmf.parallelism.jax.local_device_count", lambda: 1
    )

    num_chains, chain_method = choose_mcmc_parallelism(max_chains=2)

    assert num_chains == 2
    assert chain_method == "sequential"
