#!/bin/bash
export PYTHONPATH=".:$PYTHONPATH"
#python ./scripts/hat_cnn.py -tr_c
python ./scripts/hat_cnn.py -tr_f
python ./scripts/hat_cnn.py -tr_full
python ./scripts/hat_cnn.py -te_full
