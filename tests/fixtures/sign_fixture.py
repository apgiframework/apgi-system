"""Sign fixture for accumulated surprise.

Accumulated surprise weights the *magnitude* of each prediction error
(``S_t = Pi_e*|z_e| + Pi_i_eff*|z_i|``), so negative errors must contribute exactly as
positive ones. The fixture pins one case with both errors negative.

    Pi_i_eff = Pi_i_baseline * exp(beta * M_ca) = 1.0 * exp(1.0 * 0.2) = 1.2214
    S_t      = 1.0 * 0.8 + 1.2214 * 0.5 = 1.4107  ->  1.41
"""

from __future__ import annotations

from typing import Any, Dict

Z_E = -0.8
Z_I = -0.5
EXPECTED_S_T = 1.41
S_T_DECIMALS = 2

STATE_KWARGS: Dict[str, Any] = {
    "name": "sign-fixture",
    "description": "negative-error sign fixture",
    "phenomenology": [],
    "Pi_e_actual": 1.0,
    "Pi_i_baseline_actual": 1.0,
    "M_ca": 0.2,
    "beta": 1.0,
    "theta_t": 0.5,
}
