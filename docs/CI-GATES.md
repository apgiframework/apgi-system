# CI gates for apgi-system (register XC-CI-03)

| Register ID | Gate | Workflow job | What it checks |
|-------------|------|--------------|----------------|
| XC-CI-03.1 | Device mocks | `Acquisition gates / Device-mock gate` | Every driver and device-contract test passes against simulated devices (`-m device_mock`). |
| XC-CI-03.2 | Acquisition checks | `Acquisition gates / Acquisition checks` | `black`, `mypy` and the whole acquisition suite pass. |
| XC-CI-03.3 | Desktop builds | `Acquisition gates / Desktop build` | A PyInstaller bundle builds and its headless run records, journals, uploads and server-verifies a session. Built on Linux x86-64 only, because SQLCipher wheels exist only there. |
| XC-CI-03.4 | Offline upload | `Acquisition gates / Offline-upload gate` | The network is cut mid-upload; nothing is lost and the session resumes (`-m offline_upload`). |

Also required in practice: `Repo policy` (identifiers, README status, forbidden language, model
registry, import cycles, lockfile) and `Secret scanning`.

## Making them required

Workflow files cannot make a check *required*. In the repository settings, add a branch
protection rule (or ruleset) for `main` and `develop` that requires these status checks:

- `Device-mock gate`
- `Acquisition checks`
- `Desktop build`
- `Offline-upload gate`
- `Identifier divergence (DOI / ORCID / repo / version)`
- `gitleaks (full history)`

Until that rule exists the gates run and report but do not block a merge.
