#!/bin/bash
# Usage: bash experiments/11_advtest_firsttoken/run.sh [GPU]
# Rebuilds results/11_advtest_firsttoken/ (see experiments/11_advtest_firsttoken/README.md): CodeSearchNet python
# pretraining and AdvTest finetuning with first-token pooling and Poincare similarity, seed 42. Needs data/AdvTest/ and
# data/processed/code/python/ (the code track's process-*-code.sh python); builds data/xml/retrieval_advtest/ itself.
# Resumable: rerunning skips finished stages.
GPU=${1:-0}
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU=$GPU SEED=42 bash $SANTA/experiments/11_advtest_firsttoken/run_advtest_firsttoken.sh
