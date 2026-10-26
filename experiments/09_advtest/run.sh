#!/bin/bash
# Usage: bash experiments/09_advtest/run.sh [GPU]
# Rebuilds results/09_advtest/ (see experiments/09_advtest/README.md): CodeSearchNet python pretraining and AdvTest
# finetuning with the Poincare recipe, seed 42. Needs data/AdvTest/ and data/processed/code/python/ (the code track's
# process-*-code.sh python); builds data/xml/retrieval_advtest/ itself. Resumable: rerunning skips finished stages.
GPU=${1:-0}
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU=$GPU SEED=42 bash $SANTA/experiments/09_advtest/run_advtest_poincare.sh
