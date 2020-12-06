# Testing
This Howto explains how to test a trained model. 

To test a model we use the testing algorithm provided by TuSimple for their [Lane Detection Challenge](https://github.com/TuSimple/tusimple-benchmark).

# Add testing code
Sadly the code from TuSimple contains no license, so it can't be included in this project. To use it you have to add the code by yourself.

1. Open [tusimple-benchmark/evaluate/lane.py](https://github.com/TuSimple/tusimple-benchmark/blob/master/evaluate/lane.py) and copy all the code
2. From this project open the file src/runtime/utils/evaluation/lane.py in a text editor and paste the code from the previous step
3. Either do one of the following steps
   - **apply a patch** (provides code documentation, requires git installed)  
      The patch is bundled with this documentation. It can be downloaded below, but for the following instructions, **downloading the patch is not required**
      1. open a terminal / command line and change to your projects root directory (where `ufld.py` is located)
      2. run `git apply docs/source/_static/required_changes_for_LaneEval_+_adding_some_documentation.patch`  
      **For Windows users** this command might need some changes, at least the `/` will have to be changed to `\`.
   - **apply required fixes manually** Some fixes are required as the original code uses python 2  
      1. in line 3: change `import ujson as json` to `import json as json`
      2. in line 67: remove `or 'run_time' not in pred` (keep the `:`)  
      3. in line 71: change `run_time = pred['run_time']` to `run_time = 10`  
      4. line 93 until file end: remove this code
      5. line 56: replace `(...)` with `(json_pred, json_gt)`
      6. line 57 - 63: remove these lines

     
```{eval-rst}
:download:`required_changes_for_LaneEval_+_adding_some_documentation.patch <../_static/required_changes_for_LaneEval_+_adding_some_documentation.patch>`
```
   
