#!/bin/bash
export PYTHONPATH=".:$PYTHONPATH"
python ./scripts/resnet_baseline.py -tr -te
