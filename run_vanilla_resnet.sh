#!/bin/bash
export PYTHONPATH=".:$PYTHONPATH"
python ./scripts/vanilla_resnet.py -tr_c
python ./scripts/vanilla_resnet.py -tr_f
python ./scripts/vanilla_resnet.py -tr_full
python ./scripts/vanilla_resnet.py -te_full

