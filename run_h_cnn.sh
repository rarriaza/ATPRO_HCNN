#!/bin/bash
export PYTHONPATH=".:$PYTHONPATH"
#python ./scripts/h_cnn.py -tr_c
python ./scripts/h_cnn.py -tr_f
python ./scripts/h_cnn.py -tr_full
python ./scripts/h_cnn.py -te_full
