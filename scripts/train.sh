#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/train.sh PYTHON_ARGS

nvidia-smi
python src/train.py "$@"
