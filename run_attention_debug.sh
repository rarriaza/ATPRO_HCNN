#!/bin/bash
export PYTHONPATH=".:$PYTHONPATH"
#python ./scripts/hat_resnet.py -debug -tr_c -tr_f -tr_full -te_full
python ./scripts/hat_resnet.py -debug -te_full