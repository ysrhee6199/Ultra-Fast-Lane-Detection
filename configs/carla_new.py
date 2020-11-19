# DATA
dataset='Carla'
data_root = '/home/markus/OneDrive/Projekt_-_Fast_Lane_Detection/Datensätze/Datensatz03/'
# data_root = '/home/markus/PycharmProjects/datasets/Trainingsdatensatz1/'


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
test_txt = 'test.txt'  # default: test.txt

num_lanes = 4

# relative height of y coordinates. This is required to support different img_heights
# to access the correct h_samples for your resolution you can use something like
# [int(round(x*img_height)) for x in h_samples]
h_samples = [x/720 for x in range(160, 711, 10)]
img_height = 720
img_width = 1280




# ----
test_validation_data = 'test.json'