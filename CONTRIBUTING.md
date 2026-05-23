# Contributing

## Branch Workflow

Use `main` for stable code, `dev` for active integration, and `feature/...` branches for individual changes.

Example:

```bash
git checkout dev
git pull
git checkout -b feature/signal-diagnostics
```

## Commit Style

Use clear commit messages, for example:

```bash
git commit -m "Add signal diagnostics"
git commit -m "Refactor backtest metrics"
git commit -m "Add paper reproducibility script"
```

## Pull Requests

Open pull requests from feature branches into `dev`.

Once `dev` is stable, merge `dev` into `main`.

## Project Goals

This project is being extended into a paper comparing thematic financial signals built from TF-IDF, embeddings, and potentially EDGAR-derived text features.
