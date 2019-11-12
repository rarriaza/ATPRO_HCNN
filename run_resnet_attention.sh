#!/bin/bash
export PYTHONPATH=".:$PYTHONPATH"
python ./scripts/resnet_attention.py -tr -te
