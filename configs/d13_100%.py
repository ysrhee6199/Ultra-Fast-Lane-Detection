dataset = 'd13'
data_root = '/media/data/dataset/d13/'

train_gt = '100%_train_gt.txt'
batch_size = 16

griding_num = 100
backbone = '18'

# EXP
note = '_100%'

work_dir = '/media/data/output/'

trained_model = ''

# relative height of y coordinates. This is required to support different img_heights
# to access the correct h_samples for your resolution you can use something like
# [int(round(x*img_height)) for x in h_samples]
h_samples = [x / 720 for x in range(160, 711, 10)]
img_height = 720
img_width = 1280
