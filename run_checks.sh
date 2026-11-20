#!/usr/bin/env bash
# Validation suite for the latest built week. Exit code is non-zero on any failure.
cd "$(dirname "$0")"
pip install -q --break-system-packages pytest >/dev/null 2>&1 || true
python -m pytest -q tests "$@"
