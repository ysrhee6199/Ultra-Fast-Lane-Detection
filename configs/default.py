# DATA
dataset=None
data_root = None


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

work_dir = None

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
trained_model = None
test_work_dir = None
test_txt = 'test.txt'

num_lanes = 4

# relative height of y coordinates. This is required to support different img_heights
# to access the correct h_samples for your resolution you can use something like
# [int(round(x*img_height)) for x in h_samples]
h_samples = [x/720 for x in range(380, 711, 10)]
img_height = 720
img_width = 1280



# untested, changing these values might not work as expected. If changed use a multiple of 8
# some (possibly) relations in source code are unclear and might not be adjusted correctly
train_img_height = 288
train_img_width = 800


on_train_copy_project_to_out_dir = True

input_mode='images'
output_mode='test'
measure_time=False
video_input_file=None
camera_input_cam_number=0
screencap_recording_area = [0, 0, 1920, 1080]  # x(left), y(top), w, h
screencap_enable_image_forwarding = True # Disabling this will prevent you from using most out modules. Probably only usefull in some edge cases

video_out_enable_live_video = True
video_out_enable_video_export = False
video_out_enable_image_export = False
video_out_enable_line_mode = False



test_validation_data = 'test.json'