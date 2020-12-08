#!/bin/bash

wget https://raw.githubusercontent.com/TuSimple/tusimple-benchmark/master/evaluate/lane.py -O src/runtime/utils/evaluation/lane.py
git apply docs/source/_static/required_changes_for_LaneEval_+_adding_some_documentation.patch
