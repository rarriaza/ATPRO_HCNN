#!/bin/bash
export PYTHONPATH=".:$PYTHONPATH"
python ./scripts/h_cnn.py -tr
python ./scripts/h_cnn.py -te
