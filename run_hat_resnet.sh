#!/bin/bash
export PYTHONPATH=".:$PYTHONPATH"
python ./scripts/hat_resnet.py -tr_c
python ./scripts/hat_resnet.py -tr_f
python ./scripts/hat_resnet.py -tr_full
python ./scripts/hat_resnet.py -te_full

