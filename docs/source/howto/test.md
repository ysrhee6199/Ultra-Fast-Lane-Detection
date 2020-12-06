# Testing
This Howto explains how to test a trained model. 

To test a model we use the testing algorithm provided by TuSimple for their [Lane Detection Challenge](https://github.com/TuSimple/tusimple-benchmark).

# Add testing code
Sadly the code from TuSimple contains no license, so it can't be included in this project. To use it you have to add the code by yourself.

1. Open [tusimple-benchmark/evaluate/lane.py](https://github.com/TuSimple/tusimple-benchmark/blob/master/evaluate/lane.py) and copy all the code
2. From this project open the file src/runtime/utils/evaluation/lane.py in a text editor and paste the code from the previous step
3. Either do one of the following steps
- **apply required fixes manually** Some fixes are required as the original code uses python 2  
   in line 3: change `import ujson as json` to `import json as json`
   in line 67: remove `or 'run_time' not in pred`
   in line 71: change `run_time = pred['run_time']` to `run_time = 10`
   line 93 until file end: remove this code, it's not used in this project.
- **apply a patch** (provides code documentation)
   
