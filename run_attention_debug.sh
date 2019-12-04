#!/bin/bash
export PYTHONPATH=".:$PYTHONPATH"
#python ./scripts/resnet_attention.py -debug -tr_c -tr_f -tr_full -te_full
python ./scripts/resnet_attention.py -debug -te_full