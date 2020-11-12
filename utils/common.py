import os, argparse
from utils.dist_utils import is_main_process, dist_print, DistSummaryWriter
from utils.config import Config
import torch


def get_cfg():
    return cfg


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
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--data_root', default=None, type=str)
    parser.add_argument('--epoch', default=None, type=int)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--optimizer', default=None, type=str)
    parser.add_argument('--learning_rate', default=None, type=float)
    parser.add_argument('--weight_decay', default=None, type=float)
    parser.add_argument('--momentum', default=None, type=float)
    parser.add_argument('--scheduler', default=None, type=str)
    parser.add_argument('--steps', default=None, type=int, nargs='+')
    parser.add_argument('--gamma', default=None, type=float)
    parser.add_argument('--warmup', default=None, type=str)
    parser.add_argument('--warmup_iters', default=None, type=int)
    parser.add_argument('--backbone', default=None, type=str)
    parser.add_argument('--griding_num', default=None, type=int)
    parser.add_argument('--use_aux', default=None, type=str2bool)
    parser.add_argument('--sim_loss_w', default=None, type=float)
    parser.add_argument('--shp_loss_w', default=None, type=float)
    parser.add_argument('--note', default=None, type=str)
    parser.add_argument('--log_path', default=None, type=str)
    parser.add_argument('--finetune', default=None, type=str)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--test_model', default=None, type=str)
    parser.add_argument('--test_work_dir', default=None, type=str)
    parser.add_argument('--num_lanes', default=None, type=int)
    parser.add_argument('--train_gt', default=None, type=str)
    parser.add_argument('--cls_num_per_lane', default=None, type=int)
    parser.add_argument('--test_txt', default=None, type=str)
    parser.add_argument('--img_height', default=None, type=int)
    parser.add_argument('--img_width', default=None, type=int)
    parser.add_argument('--train_img_height', default=None, type=int)
    parser.add_argument('--train_img_width', default=None, type=int)
    parser.add_argument('--test_validation_data', default=None, type=str)
    parser.add_argument('--output_mode', default=None, type=str,
                        help='only applicable for output.py, specifies output module')
    parser.add_argument('--input_mode', default=None, type=str,
                        help='only applicable for output.py, specifies input module')
    return parser


def merge_config():
    args = get_args().parse_args()
    user_cfg = Config.fromfile(args.config)
    cfg = Config.fromfile('configs/default.py')

    for item in [(k, v) for k, v in user_cfg.items() if v]:
        cfg[item[0]] = item[1]

    items = ['dataset', 'data_root', 'epoch', 'batch_size', 'optimizer', 'learning_rate',
             'weight_decay', 'momentum', 'scheduler', 'steps', 'gamma', 'warmup', 'warmup_iters',
             'use_aux', 'griding_num', 'backbone', 'sim_loss_w', 'shp_loss_w', 'note', 'log_path',
             'finetune', 'resume', 'test_model', 'test_work_dir', 'num_lanes', 'train_gt', 'cls_num_per_lane',
             'test_txt', 'output_mode', 'input_mode', 'test_validation_data']
    # following cfgs cant be set via cli: 'h_samples', 'test_splits'

    for item in items:
        if getattr(args, item) is not None:
            dist_print('merge ', item, ' config')
            setattr(cfg, item, getattr(args, item))
    return args, cfg


# TODO: delete old iterations
def save_model(net, optimizer, epoch, save_path, distributed):
    safe_distance = 10
    if epoch % safe_distance == 0:
        if is_main_process():
            model_state_dict = net.state_dict()
            state = {'model': model_state_dict, 'optimizer': optimizer.state_dict()}
            # state = {'model': model_state_dict}
            assert os.path.exists(save_path)
            model_path = os.path.join(save_path, 'ep%03d.pth' % epoch)
            torch.save(state, model_path)

            # delete old epochs
            if epoch > 0:
                prev_model_path = os.path.join(save_path, 'ep%03d.pth' % (int(epoch)-safe_distance))
                if os.path.exists(prev_model_path):
                    os.remove(prev_model_path)


import pathspec


# TODO: copying files with spaces in their path seems to fail
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
        # pdb.set_trace()
        for f in to_cp_files:
            dirs = os.path.join(to_path, 'code', os.path.split(f[2:])[0])
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            os.system('cp %s %s' % (f, os.path.join(to_path, 'code', f[2:])))


import datetime, os


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
