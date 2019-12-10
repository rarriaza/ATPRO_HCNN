#!/bin/bash
export PYTHONPATH=".:$PYTHONPATH"
python ./scripts/vanilla_resnet.py -tr
python ./scripts/vanilla_resnet.py -te

