"""
Regression tests for model default arguments in panel_nmf_model.model().

Covers to-do item 1: the missing-mask default must mean "nothing is missing"
when missing_idx_array is not provided.

These tests verify the mask construction logic and its downstream effect on
observation selection, without running the full NumPyro plate-based model
(which requires realistic dimension sizes for broadcasting).
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: extract the mask logic from panel_nmf_model.model()
# ---------------------------------------------------------------------------


def _build_missing_idx(denominators, missing_idx_array):
    """Replicate the missing-mask default logic from model() lines 38-41."""
    if missing_idx_array is None:
        # This is the line that was fixed (was np.ones_like → np.zeros_like)
        missing_idx = np.zeros_like(denominators, dtype=np.bool_).reshape(-1)
    else:
        missing_idx = missing_idx_array.reshape(-1)
    return missing_idx


def _build_control_idx(denominators, control_idx_array):
    """Replicate the control-mask default logic from model() lines 31-36."""
    if control_idx_array is None:
        control_idx = np.ones_like(denominators, dtype=np.bool_).reshape(-1)
    else:
        control_idx = control_idx_array.reshape(-1)
    return control_idx


def _compute_observation_mask(missing_idx, control_idx, model_treated):
    """Replicate the observation mask logic from model() lines 156-159."""
    if model_treated:
        mask = ~missing_idx
    else:
        mask = ~missing_idx & control_idx
    return mask


# ---------------------------------------------------------------------------
# Tests: missing mask default (to-do item 1)
# ---------------------------------------------------------------------------


class TestMissingMaskDefault:
    """To-do item 1: missing_idx_array=None must mean 'nothing is missing'."""

    @pytest.mark.parametrize("shape", [(1, 2, 3), (4, 50, 48), (1, 1, 1)])
    def test_none_produces_all_false(self, shape):
        """When missing_idx_array=None, the mask should be all-False."""
        denominators = np.ones(shape)
        missing_idx = _build_missing_idx(denominators, None)
        assert not missing_idx.any(), (
            "missing_idx_array=None should produce all-False (nothing is missing)"
        )

    @pytest.mark.parametrize("shape", [(1, 2, 3), (4, 50, 48)])
    def test_none_equivalent_to_explicit_all_false(self, shape):
        """None should behave identically to an explicit all-False mask."""
        denominators = np.ones(shape)
        explicit_mask = np.zeros(shape, dtype=bool)

        from_none = _build_missing_idx(denominators, None)
        from_explicit = _build_missing_idx(denominators, explicit_mask)

        np.testing.assert_array_equal(from_none, from_explicit)

    def test_none_includes_all_observations_in_likelihood(self):
        """With nothing missing and all-control, every observation should enter the likelihood."""
        shape = (2, 5, 10)
        denominators = np.ones(shape)

        missing_idx = _build_missing_idx(denominators, None)
        control_idx = _build_control_idx(denominators, None)
        mask = _compute_observation_mask(missing_idx, control_idx, model_treated=False)

        assert mask.sum() == np.prod(shape), (
            f"Expected all {np.prod(shape)} observations in likelihood, "
            f"got {mask.sum()}"
        )

    def test_old_bug_all_ones_excludes_everything(self):
        """Document the old bug: np.ones_like (all-True) masks out every observation.

        Before the fix, missing_idx_array=None produced an all-True mask,
        meaning 'everything is missing'. Then mask = ~missing_idx & control_idx
        becomes all-False, excluding every observation from the likelihood.
        """
        shape = (2, 5, 10)
        denominators = np.ones(shape)

        # Simulate OLD buggy default
        buggy_missing = np.ones_like(denominators, dtype=np.bool_).reshape(-1)
        control_idx = _build_control_idx(denominators, None)
        mask = _compute_observation_mask(
            buggy_missing, control_idx, model_treated=False
        )

        assert not mask.any(), (
            "All-True missing mask should exclude every observation — "
            "this was the old (buggy) default."
        )

    def test_partial_missing_mask_excludes_correct_subset(self):
        """A partial missing mask should only exclude the marked observations."""
        shape = (1, 3, 4)  # 12 total observations
        denominators = np.ones(shape)

        missing = np.zeros(shape, dtype=bool)
        missing[0, 0, :] = True  # mark first unit as fully missing (4 obs)

        missing_idx = _build_missing_idx(denominators, missing)
        control_idx = _build_control_idx(denominators, None)
        mask = _compute_observation_mask(missing_idx, control_idx, model_treated=False)

        assert mask.sum() == 8, f"Expected 8 non-missing observations, got {mask.sum()}"


# ---------------------------------------------------------------------------
# Tests: control mask default
# ---------------------------------------------------------------------------


class TestControlIdxDefault:
    """Verify that control_idx_array=None means 'everything is control'."""

    def test_none_produces_all_true(self):
        denominators = np.ones((2, 5, 10))
        control_idx = _build_control_idx(denominators, None)
        assert control_idx.all(), (
            "control_idx_array=None should mean everything is control (all-True)"
        )

    def test_none_with_model_treated_false_includes_all(self):
        """With both masks at None and model_treated=False, all obs enter the likelihood."""
        shape = (2, 5, 10)
        denominators = np.ones(shape)

        missing_idx = _build_missing_idx(denominators, None)
        control_idx = _build_control_idx(denominators, None)
        mask = _compute_observation_mask(missing_idx, control_idx, model_treated=False)

        assert mask.sum() == np.prod(shape)

    def test_control_mask_excludes_treated_when_not_modeling_treatment(self):
        """When model_treated=False, treated observations should be excluded."""
        shape = (1, 2, 4)
        denominators = np.ones(shape)

        control = np.ones(shape, dtype=bool)
        control[0, :, 2:] = False  # last 2 time periods are treated

        missing_idx = _build_missing_idx(denominators, None)
        control_idx = _build_control_idx(denominators, control)
        mask = _compute_observation_mask(missing_idx, control_idx, model_treated=False)

        # 2 units x 2 control periods = 4 observations
        assert mask.sum() == 4

    def test_model_treated_ignores_control_mask(self):
        """When model_treated=True, control_idx doesn't affect the mask."""
        shape = (1, 2, 4)
        denominators = np.ones(shape)

        control = np.ones(shape, dtype=bool)
        control[0, :, 2:] = False  # last 2 time periods are treated

        missing_idx = _build_missing_idx(denominators, None)
        mask = _compute_observation_mask(
            missing_idx, control.reshape(-1), model_treated=True
        )

        # model_treated=True: mask = ~missing_idx, ignoring control
        assert mask.sum() == np.prod(shape)
