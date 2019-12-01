#!/bin/bash
export PYTHONPATH=".:$PYTHONPATH"
python ./scripts/resnet_attention.py -tr_c
python ./scripts/resnet_attention.py -tr_f
python ./scripts/resnet_attention.py -te

