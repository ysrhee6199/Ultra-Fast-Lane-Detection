import argparse
from utils.dist_utils import is_main_process, dist_print, DistSummaryWriter
from utils.config import Config
import datetime, os
import torch
import pathspec


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, argument_default=None)
    parser.description = """Most config values can be overwritten with its corresponding cli parameters. For further details on the config options see configs/default.py and the documentation.
unavailable options on cli:
- h_samples
"""

    # @formatter:off
    # define config groups, only relevant for --help
    basics = parser.add_argument_group('basic switches, these are always needed')
    train_args = parser.add_argument_group('training', 'these switches are only used for training')
    runtime_args = parser.add_argument_group('runtime', 'these switches are only used in the runtime module')
    in_modules = parser.add_argument_group('input modules', 'with these options you can configure the input modules. Each module may have its own config switches')
    out_modules = parser.add_argument_group('output modules', 'with these options you can configure the output modules. Each module may have its own config switches')

    # define switches
    basics.add_argument('config', help='path to config file')
    basics.add_argument('--dataset', metavar='', type=str, help='dataset name, can be any string')
    basics.add_argument('--data_root', metavar='', type=str, help='root directory of your dataset')
    basics.add_argument('--batch_size', metavar='', type=int, help='size of samples dataloader will load for each batch')
    basics.add_argument('--backbone', metavar='', type=str, help="define which resnet backbone to use, allowed values: ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']")
    basics.add_argument('--griding_num', metavar='', type=int, help='x resolution of nn, just like h_samples are the y resolution')
    basics.add_argument('--note', metavar='', type=str, help='suffix for working directory (probably good to give them a rememberable name')
    basics.add_argument('--log_path', metavar='', type=str, help='working directory: every output will be written here')
    basics.add_argument('--use_aux', metavar='', type=str2bool, help='used to improve training, should be disabled during runtime (independent of this config)')
    basics.add_argument('--num_lanes', metavar='', type=int, help='number of lanes')
    basics.add_argument('--img_height', metavar='', type=int, help='height of input images')
    basics.add_argument('--img_width', metavar='', type=int, help='width of input images')
    basics.add_argument('--train_img_height', metavar='', type=int, help='height of image which will be passed to nn; !this option is untested and might not work!')
    basics.add_argument('--train_img_width', metavar='', type=int, help='width of image which will be passed to nn; !this option is untested and might not work!')

    train_args.add_argument('--local_rank', metavar='', type=int, default=0)
    train_args.add_argument('--epoch', metavar='', type=int, help='number of epochs to train')
    train_args.add_argument('--optimizer', metavar='', type=str)
    train_args.add_argument('--learning_rate', metavar='', type=float)
    train_args.add_argument('--weight_decay', metavar='', type=float)
    train_args.add_argument('--momentum', metavar='', type=float)
    train_args.add_argument('--scheduler', metavar='', type=str)
    train_args.add_argument('--steps', metavar='', type=int, nargs='+')
    train_args.add_argument('--gamma', metavar='', type=float)
    train_args.add_argument('--warmup', metavar='', type=str)
    train_args.add_argument('--warmup_iters', metavar='', type=int)
    train_args.add_argument('--sim_loss_w', metavar='', type=float)
    train_args.add_argument('--shp_loss_w', metavar='', type=float)
    train_args.add_argument('--finetune', metavar='', type=str)
    train_args.add_argument('--resume', metavar='', type=str, help='path of existing model; continue training this model')
    train_args.add_argument('--train_gt', metavar='', type=str, help='training index file (train_gt.txt)')
    train_args.add_argument('--on_train_copy_project_to_out_dir', metavar='', type=str2bool, help='define whether the project project directory is copied to the output directory')

    runtime_args.add_argument('--test_model', metavar='', type=str, help='load trained model and use it for evaluation')
    runtime_args.add_argument('--output_mode', metavar='', type=str, help='only applicable for output.py, specifies output module')
    runtime_args.add_argument('--input_mode', metavar='', type=str, help='only applicable for output.py, specifies input module')
    runtime_args.add_argument('--measure_time', metavar='', type=str2bool, help='enable speed measurement')

    in_modules.add_argument('--video_input_file', metavar='', type=str, help='full filepath to video file you want to use as input')
    in_modules.add_argument('--camera_input_cam_number', metavar='', type=int, help='number of your camera')
    in_modules.add_argument('--screencap_recording_area', metavar='', type=int, nargs='+', help='position and size of recording area: x,y,w,h')

    out_modules.add_argument('--test_txt', metavar='', type=str, help='testing index file (test.txt)')
    out_modules.add_argument('--test_validation_data', metavar='', type=str, help='file containing labels for test data to validate test results')
    # @formatter:on
    return parser


def merge_config() -> Config:
    """ combines default and user-config and cli arguments
    """
    args = get_args().parse_args()
    user_cfg = Config.fromfile(args.config)
    cfg = Config.fromfile('configs/default.py')

    # override default cfg with values from user cfg
    for item in [(k, v) for k, v in user_cfg.items() if v]:
        cfg[item[0]] = item[1]

    for k,v in vars(args).items():
        if v:
            dist_print('merge ', (k,v), ' config')
            setattr(cfg, k, v)
    return cfg


def save_model(net, optimizer, epoch, save_path, distributed):
    if is_main_process():
        model_state_dict = net.state_dict()
        state = {'model': model_state_dict, 'optimizer': optimizer.state_dict()}
        # state = {'model': model_state_dict}
        assert os.path.exists(save_path)
        model_path = os.path.join(save_path, 'ep%03d.pth' % epoch)
        torch.save(state, model_path)



def cp_projects(to_path):
    if is_main_process():
        with open('./.gitignore', 'r') as fp:
            ign = fp.read()
        ign += '\n.git'
        spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, ign.splitlines())
        all_files = {os.path.join(root, name) for root, dirs, files in os.walk('./') for name in files}
        matches = spec.match_files(all_files)
        matches = set(matches)
        to_cp_files = all_files - matches
        # to_cp_files = [f[2:] for f in to_cp_files]
        for f in to_cp_files:
            dirs = os.path.join(to_path, 'code', os.path.split(f[2:])[0])
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            os.system('cp "%s" "%s"' % (f, os.path.join(to_path, 'code', f[2:])))


def get_work_dir(cfg):
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    hyper_param_str = '_lr_%1.0e_b_%d' % (cfg.learning_rate, cfg.batch_size)
    work_dir = os.path.join(cfg.log_path, now + hyper_param_str + cfg.note)
    return work_dir


def get_logger(work_dir, cfg):
    logger = DistSummaryWriter(work_dir)
    config_txt = os.path.join(work_dir, 'cfg.txt')
    if is_main_process():
        with open(config_txt, 'w') as fp:
            fp.write(str(cfg))

    return logger
