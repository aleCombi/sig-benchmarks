# sig-benchmarks

Minimal benchmarks for signature libraries.

## Setup

```bash
uv init
cp pyproject.toml .
uv sync
```

## Run

```bash
uv run benchmark.py
```

## What it tests

- **Speed:** chen vs iisignature vs pysiglib
- **Correctness:** All three implementations
- **Gradients:** chen vs pysiglib (PyTorch)
