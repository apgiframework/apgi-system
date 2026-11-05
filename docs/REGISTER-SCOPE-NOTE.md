# Scope note: attached task registers vs. this repository

The task registers supplied for this change set describe three different codebases:

| Attached file(s) | Describes | Applies to apgi-system? |
|---|---|---|
| `system-all-tasks.csv`, `system-still-open-items.csv` | `apgi-api` (517 / 499 rows; Repository column is `apgiframework/apgi-api`) | No. Code paths such as `app/services/audit_signing.py` do not exist here. |
| `api-all-tasks.csv`, `api-still-open-items.csv` | `apgi-api` audit rows (S1–S7, v1.2 compliance) | No, except the cross-repo S6-SYS-31 signing/SHA-256 integration, whose counterpart is in apgi-api. |
| `version-12-requirements.csv` | The A05 validation application (`apgi_core/`, `protocols/`, `Validation_GUI.py`) | No. None of those paths exist in this checkout. |
| `Software-apgi-system.docx` | apgi-system | Yes. See "Still to implement" below. |

## apgi-system items from the audit

- Coverage floor: `ci-cd.yml` sets `COVERAGE_FLOOR=0`. The README claim of a 90% floor was stale and has been corrected. Setting a real floor needs a measured baseline first (open).
- Benchmarks and dependency audit remain advisory (`continue-on-error`); open.
- macOS unit-test matrix and `develop` gate scope (XC-CI-03): open, CI/repo-settings work.
- S6-SYS-31 apgi-api routes and private-bucket V4 signing: must be done in apgi-api.
- Hardware (S6-SYS-02/06/07/08/22.6/22.12), native Windows/macOS packaging, production keychain, external bids-validator: need devices, platforms or decisions (D10, S6-DEC-01); not implementable here.
