dataset = 'carla'
data_root = '/home/nvidia/Carla-Lane-Detection-Dataset-Generation/src/data/dataset/Town04_Opt/'

train_gt = 'train_gt.txt'
batch_size = 16

backbone = '34'
num_lanes = 4
# EXP
note = 'carla'

work_dir = '/home/nvidia/Carla-Lane-Detection-Dataset-Generation/src/'

# relative height of y coordinates. This is required to support different img_heights
# to access the correct h_samples for your resolution you can use something like
# [int(round(x*img_height)) for x in h_samples]
h_samples = [x / 480 for x in range(210, 471, 10)]
img_height = 480
img_width = 640
