# Changelog

All notable changes to `apgi-system` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-10-17

Brings the system into compliance with APGI protocol specification 1.1.0, aligned with
`apgi-framework` 1.1.0.

- `protocols/` (new): all eight protocol JSONs (Protocol 0–7), the Draft 2020-12 schema,
  `CHANGELOG.md`, `MANIFEST.json` and `VALIDATION.json`, byte-identical to apgi-framework 1.1.0.
- `scripts/validate_protocols.py` and `tests/test_validate_protocols.py`, taken unchanged from
  apgi-framework. Run with `make protocols-check`; CI runs it before the unit tests. `jsonschema`
  joins the `[dev]` extras and `requirements.txt`.
- `apgi_framework/config/protocol_spec.py` (new): loads the registered protocols (version,
  study status, cohort sizes, per-prediction status and thresholds, the reduced empirical model)
  and adds `Verdict` / `classify_prediction`, the v1.1 rule that a failure to confirm is
  INDETERMINATE, not FALSIFIED. Exposes the registered θ₀, γ_sig and β_SM ranges; κ stays `null`
  (uncalibrated) and has no legacy default.
- `statistical_report_generator`: the `soma_bias_absence` criterion returned FALSIFIED when β ≈ 1
  and p > 0.05. Nonsignificance does not establish equivalence, so it now returns INCONCLUSIVE.
- `falsification/soma_bias.py`: a nonsignificant bias test no longer reads "framework supported";
  it is reported as indeterminate.
- Version 1.1.0. `apgi_framework/__version__.py`, `__init__.py` and the stale `_version.py`
  (was 0.1.0) all say 1.1.0, and a test fails if they drift apart. The API's version comes from
  `api.__version__` via `settings.api_version`, used by the FastAPI app, `GET /` and
  `/v1/version` (`API_VERSION` still overrides).
- Project name unified as *Allostatic Precision-Gated Ignition*. It was previously also given as
  "Interoceptive Predictive Integration", "Active Predictive Global Ignition", "Active Predictive
  Generative Inference/Intelligence", "Active Precision Gating and Interoception", "Active
  Perception and Global Integration" and "Adaptive Precision and Generalized Intelligence".
- `CITATION.cff` (new) and a BibTeX entry in `README.md`, both 1.1.0.
- `tests/test_protocol_spec_v1_1.py` (new): release, κ, cohort, verdict and version-agreement
  checks.

## [1.0.0] - 2026-01-23

Initial release.
