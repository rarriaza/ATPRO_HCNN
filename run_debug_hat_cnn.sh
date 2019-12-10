#!/bin/bash
export PYTHONPATH=".:$PYTHONPATH"
python ./scripts/hat_cnn.py -debug -tr_c
python ./scripts/hat_cnn.py -debug -tr_f
python ./scripts/hat_cnn.py -debug -tr_full
python ./scripts/hat_cnn.py -debug -te_full
