#!/bin/sh
# sh scripts/test_all_models_in_dir.sh <path to config> <path to model directory>
for model in $2/*.pth;
do
  python ufld.py $1 --mode=test --trained_model $model --batch_size=4;
done