import json
import os

from evaluation.eval_wrapper import run_test_tusimple
from evaluation.tusimple.lane import LaneEval
from utils.global_config import cfg


def simplified_combine_tusimple_test(work_dir, exp_name):
    all_res = []
    output_path = os.path.join(work_dir, exp_name + '.txt')
    with open(output_path, 'r') as fp:
        res = fp.readlines()
    all_res.extend(res)
    names = set()
    all_res_no_dup = []
    for i, res in enumerate(all_res):
        pos = res.find('clips')
        name = res[pos:].split('\"')[0]
        if name not in names:
            names.add(name)
            all_res_no_dup.append(res)

    output_path = os.path.join(work_dir, exp_name + '.txt')
    with open(output_path, 'w') as fp:
        fp.writelines(all_res_no_dup)


#
def simplified_eval_lane(net, dataset, data_root, work_dir, griding_num, use_aux):
    net.eval()

    if dataset == 'CULane':
        raise NotImplemented
    elif dataset == 'Carla' or dataset == 'Tusimple':
        exp_name = f'{dataset}_eval_tmp'
        # this line does the nn evaluation
        run_test_tusimple(net, data_root, work_dir, exp_name, griding_num, use_aux, distributed=False)

        simplified_combine_tusimple_test(work_dir, exp_name)

        # probably evaluates quality of one dataset
        res = LaneEval.bench_one_submit(os.path.join(work_dir, exp_name + '.txt'),
                                        os.path.join(data_root, cfg.test_validation_data))
        res = json.loads(res)
        for r in res:
            print(r['name'], r['value'])


def out_test(predictions, names):
    # this code is .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... .... simply wrong xD
    return
    net = None
    if not os.path.exists(cfg.test_work_dir):
        os.mkdir(cfg.test_work_dir)

    simplified_eval_lane(net, cfg.dataset, cfg.data_root, cfg.test_work_dir, cfg.griding_num, False, distributed=False)
