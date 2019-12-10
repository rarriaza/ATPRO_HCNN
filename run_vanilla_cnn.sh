#!/bin/bash
export PYTHONPATH=".:$PYTHONPATH"
python ./scripts/vanilla_cnn.py -tr
python ./scripts/vanilla_cnn.py -te
