# This configuration contains many params which are redundant with the default configuration
# but i'll keep this version as it is tested and verified to work.

# DATA
dataset = 'Tusimple'
data_root = '/media/data/tusimple'

# TRAIN
train_gt = 'train_gt.txt'
epoch = 100
batch_size = 4
optimizer = 'Adam'  # ['SGD','Adam']
# learning_rate = 0.1
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'  # ['multi', 'cos']
# steps = [50,75]
gamma = 0.1
warmup = 'linear'
warmup_iters = 100

# NETWORK
backbone = '18'
griding_num = 100
use_aux = True

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0

# EXP
note = ''

work_dir = '/media/data/output/'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
trained_model = None
test_work_dir = None
test_txt = None  # default: test.txt


num_lanes = 4

# relative height of y coordinates. This is required to support different img_heights
# to access the correct h_samples for your resolution you can use something like
# [int(round(x*img_height)) for x in h_samples]
h_samples = [x/720 for x in range(160, 711, 10)]
img_height = 720
img_width = 1280
