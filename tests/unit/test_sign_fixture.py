"""Accumulated surprise ignores the sign of the prediction errors (z_e = -0.8, z_i = -0.5 -> 1.41)."""

from __future__ import annotations

import itertools

import pytest

from APGI_Equations import PsychologicalState, StateCategory
from tests.fixtures.sign_fixture import EXPECTED_S_T, S_T_DECIMALS, STATE_KWARGS, Z_E, Z_I


def make(z_e: float, z_i: float) -> PsychologicalState:
    return PsychologicalState(category=list(StateCategory)[0], z_e=z_e, z_i=z_i, **STATE_KWARGS)


def test_negative_errors_give_the_fixture_value() -> None:
    assert round(make(Z_E, Z_I).S_t, S_T_DECIMALS) == EXPECTED_S_T


@pytest.mark.parametrize("sign_e,sign_i", list(itertools.product((1, -1), repeat=2)))
def test_every_sign_combination_gives_the_same_surprise(sign_e: int, sign_i: int) -> None:
    state = make(sign_e * abs(Z_E), sign_i * abs(Z_I))
    assert state.S_t == pytest.approx(make(Z_E, Z_I).S_t)
    assert round(state.S_t, S_T_DECIMALS) == EXPECTED_S_T


def test_fixture_value_is_not_the_signed_sum() -> None:
    signed = 1.0 * Z_E + make(Z_E, Z_I).Pi_i_eff_actual * Z_I
    assert signed < 0 and round(signed, S_T_DECIMALS) != EXPECTED_S_T


def test_dynamical_inputs_keep_the_raw_error_signs() -> None:
    inputs = make(Z_E, Z_I).to_dynamical_inputs(time=0.0)
    assert inputs["eps_e"] == pytest.approx(Z_E) and inputs["eps_i"] == pytest.approx(Z_I)
