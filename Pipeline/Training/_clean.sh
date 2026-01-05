#!/usr/bin/env bash
# Deletes everything inside the "Logs" and "Checkpoints" directories

set -euo pipefail

# 1. Clear the Logs directory (only its contents)
if [ -d "../Logs" ]; then
  echo "Cleaning contents of Logs/ directory..."
  rm -rf ../Logs
else
  echo "Warning: Logs directory does not exist."
fi

# 2. Clear the Checkpoints directory (only its contents)
if [ -d "../Checkpoints" ]; then
  echo "Cleaning contents of Checkpoints/ directory..."
  rm -rf ../Checkpoints
else
  echo "Warning: Checkpoints directory does not exist."
fi