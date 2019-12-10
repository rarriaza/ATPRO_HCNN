#!/bin/bash
export PYTHONPATH=".:$PYTHONPATH"
python ./scripts/h_resnet.py -tr_c
python ./scripts/h_resnet.py -tr_f
python ./scripts/h_resnet.py -tr_full
python ./scripts/h_resnet.py -te_full

