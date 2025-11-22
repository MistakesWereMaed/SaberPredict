#!/usr/bin/env bash
# Run all models
python trainer.py --model MLP
python trainer.py --model GNN
python trainer.py --model TCN
# Cleanup
rm -f .lr_find*