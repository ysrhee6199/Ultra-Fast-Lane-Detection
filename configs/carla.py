# DATA
dataset='Carla'
data_root = '/home/markus/PycharmProjects/datasets/Trainingsdatensatz1/'
cls_num_per_lane = 34   # number of h_samples; not determining automatically because only available during training (test). TODO: should be determined automatically and stored together with model

# TRAIN
train_gt = 'train_gt.txt'
epoch = 100
batch_size = 4
optimizer = 'Adam'  #['SGD','Adam']
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos' #['multi', 'cos']
steps = [25,38]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 100

# NETWORK
griding_num = 100
backbone = '18'
use_aux = True

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = '/home/markus/PycharmProjects/log/carla/'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = None
test_work_dir = None
test_txt = None  # default: test.txt
test_splits = ['train.txt']  # for demo.py

num_lanes = 4


h_samples = [380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
img_height = 720
img_width = 1280
train_img_height = 288  # default: 288
train_img_width = 800  # default: 800
