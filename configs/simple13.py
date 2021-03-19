dataset = 'simple13'
data_root = '/media/data/dataset/simple13/'

train_gt = 'train_gt.txt'
batch_size = 64

griding_num = 100
epoch = 100
backbone = '18'

# EXP
note = '_simple13_equal'

work_dir = '/media/data/output/'

trained_model = ''

# relative height of y coordinates. This is required to support different img_heights
# to access the correct h_samples for your resolution you can use something like
# [int(round(x*img_height)) for x in h_samples]
h_samples = [x / 720 for x in range(160, 711, 10)]
img_height = 720
img_width = 1280
