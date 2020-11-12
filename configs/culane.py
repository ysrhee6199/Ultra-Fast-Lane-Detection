# DATA
dataset = 'CULane'
data_root = None

# TRAIN
train_gt = 'list/train_gt.txt'
epoch = 50
batch_size = 32
optimizer = 'SGD'  # ['SGD','Adam']
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.9

scheduler = 'multi'  # ['multi', 'cos']
steps = [25, 38]
gamma = 0.1
warmup = 'linear'
warmup_iters = 695

# NETWORK
use_aux = True
griding_num = 200
backbone = '18'

# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = None

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = None
test_work_dir = None
test_txt = 'list/test.txt'  # default: test.txt
test_splits = ['list/test_split/'+split for split in ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']]  # for demo.py

num_lanes = 4

h_samples = [x for x in range(290, 591, 10)]
img_height = 590
img_width = 1640

cls_num_per_lane = len(h_samples)   # number of h_samples; not determining automatically because only available during training (test). TODO: should be determined automatically and stored together with model