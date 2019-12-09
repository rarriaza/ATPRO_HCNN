#!/bin/bash
export PYTHONPATH=".:$PYTHONPATH"
python ./scripts/baseline_architecture.py -tr_c
python ./scripts/baseline_architecture.py -tr_f
python ./scripts/baseline_architecture.py -tr_full
python ./scripts/baseline_architecture.py -te_full

