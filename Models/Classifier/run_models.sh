#!/usr/bin/env bash
python ./Scripts/train.py --model MLP
python ./Scripts/train.py --model GNN
python ./Scripts/train.py --model TCN