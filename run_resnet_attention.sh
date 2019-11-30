#!/bin/bash
export PYTHONPATH=".:$PYTHONPATH"
python ./scripts/resnet_attention.py -tr_c -tr_f -te
