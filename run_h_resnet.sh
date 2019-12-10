#!/bin/bash
export PYTHONPATH=".:$PYTHONPATH"
python ./scripts/vanilla_resnet.py -tr -te
