# Project-integrated Alphalens-Reloaded diagnostics

This folder contains the Alphalens-Reloaded comparison layer.

The custom `src/signal_diagnostics.py` module remains the baseline for the
paper because it is more transparent.

This layer connects the same project data pipeline to Alphalens:

```text
src/config.py
src/data_ingest.py
src/signal_methods.py
src/alpha_model.py
src/signal_diagnostics_alphalens.py
```

## Files

```text
src/signal_diagnostics_alphalens.py
tests/test_signal_diagnostics_alphalens.py
notebooks/04d_alphalens_reloaded_project_integrated.ipynb
pytest.ini
```

## Install Alphalens-Reloaded

```bash
pip install alphalens-reloaded
```

## Run tests

```bash
pytest -q
```

Most tests run without Alphalens installed because the adapter keeps a simple
fallback path. The final smoke test is skipped unless `alphalens-reloaded` is
installed.
