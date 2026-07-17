"""Characterization tests for run_mcmc_inference / generate_predictions
config-defaulting logic, before extracting the shared outcome_dist/
nb_disp/sample_disp resolution into _resolve_model_settings (Tier 3 of
the repo clarity refactor)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bayesian_panel_nmf.inference import generate_predictions, run_mcmc_inference


def _minimal_data_dict():
    shape = (1, 1, 1)
    return {
        "Y": np.ones(shape),
        "denominators": np.ones(shape),
        "control_idx_array": np.ones(shape, dtype=bool),
        "missing_idx_array": np.zeros(shape, dtype=bool),
        # validate_data_dict also requires these three metadata keys.
        "groups": ["g"],
        "units": ["u"],
        "times": ["2020-01-01"],
    }


def test_run_mcmc_inference_defaults_outcome_dist_to_nb_when_absent(
    make_inference_config,
):
    data_dict = _minimal_data_dict()
    config = make_inference_config(mcmc={"num_chains": 1})

    with patch("bayesian_panel_nmf.inference.MCMC") as mock_mcmc_cls:
        mock_mcmc = MagicMock()
        mock_mcmc.get_samples.return_value = {"mu": np.zeros((1,))}
        mock_mcmc_cls.return_value = mock_mcmc

        run_mcmc_inference(
            data_dict, model_fn=lambda **kwargs: None, rank=1, config=config
        )

    call_kwargs = mock_mcmc.run.call_args.kwargs
    assert call_kwargs["outcome_dist"] == "NB"
    assert call_kwargs["nb_disp"] == pytest.approx(1e-4)
    assert call_kwargs["sample_disp"] is False


def test_run_mcmc_inference_uses_configured_outcome_dist_when_present(
    make_inference_config,
):
    data_dict = _minimal_data_dict()
    config = make_inference_config(
        mcmc={"num_chains": 1},
        model={
            "outcome_distribution": "Poisson",
            "nb_disp": 0.5,
            "sample_disp": True,
        },
    )

    with patch("bayesian_panel_nmf.inference.MCMC") as mock_mcmc_cls:
        mock_mcmc = MagicMock()
        mock_mcmc.get_samples.return_value = {"mu": np.zeros((1,))}
        mock_mcmc_cls.return_value = mock_mcmc

        run_mcmc_inference(
            data_dict, model_fn=lambda **kwargs: None, rank=1, config=config
        )

    call_kwargs = mock_mcmc.run.call_args.kwargs
    assert call_kwargs["outcome_dist"] == "Poisson"
    assert call_kwargs["nb_disp"] == pytest.approx(0.5)
    assert call_kwargs["sample_disp"] is True


def test_generate_predictions_defaults_outcome_dist_to_nb_when_absent(
    make_inference_config,
):
    data_dict = {"denominators": np.ones((1, 1, 1))}
    config = make_inference_config()

    mock_mcmc = MagicMock()
    mock_mcmc.get_samples.return_value = {"mu": np.zeros((1,))}
    mock_mcmc.num_chains = 1

    with patch("bayesian_panel_nmf.inference.Predictive") as mock_predictive_cls:
        mock_predictive = MagicMock(return_value={"y_obs": np.zeros((1, 1, 1, 1))})
        mock_predictive_cls.return_value = mock_predictive

        generate_predictions(
            mock_mcmc, data_dict, model_fn=lambda **kwargs: None, rank=1, config=config
        )

    call_kwargs = mock_predictive.call_args.kwargs
    assert call_kwargs["outcome_dist"] == "NB"
    assert call_kwargs["nb_disp"] == pytest.approx(1e-4)
    assert call_kwargs["sample_disp"] is False


def test_generate_predictions_uses_configured_outcome_dist_when_present(
    make_inference_config,
):
    data_dict = {"denominators": np.ones((1, 1, 1))}
    config = make_inference_config(
        model={
            "outcome_distribution": "Poisson",
            "nb_disp": 0.5,
            "sample_disp": True,
        },
    )

    mock_mcmc = MagicMock()
    mock_mcmc.get_samples.return_value = {"mu": np.zeros((1,))}
    mock_mcmc.num_chains = 1

    with patch("bayesian_panel_nmf.inference.Predictive") as mock_predictive_cls:
        mock_predictive = MagicMock(return_value={"y_obs": np.zeros((1, 1, 1, 1))})
        mock_predictive_cls.return_value = mock_predictive

        generate_predictions(
            mock_mcmc, data_dict, model_fn=lambda **kwargs: None, rank=1, config=config
        )

    call_kwargs = mock_predictive.call_args.kwargs
    assert call_kwargs["outcome_dist"] == "Poisson"
    assert call_kwargs["nb_disp"] == pytest.approx(0.5)
    assert call_kwargs["sample_disp"] is True
