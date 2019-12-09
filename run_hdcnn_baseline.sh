#!/bin/bash
export PYTHONPATH=".:$PYTHONPATH"
python ./scripts/hdcnn_baseline.py -tr -te
