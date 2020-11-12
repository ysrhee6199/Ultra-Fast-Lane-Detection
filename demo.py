import os, cv2
from model.model import parsingNet
from utils import global_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import gen_row_anchor, default_img_transforms

if __name__ == "__main__":
    args = global_config.args
    cfg = global_config.cfg
    torch.backends.cudnn.benchmark = True

    dist_print('start testing...')
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    net = parsingNet(
        pretrained=False,
        backbone=cfg.backbone,
        cls_dim=(cfg.griding_num + 1, cfg.cls_num_per_lane, cfg.num_lanes),
        use_aux=False
    ).cuda()  # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)




    net.eval()

    img_transforms = default_img_transforms
    splits = cfg.test_splits
    datasets = [LaneTestDataset(cfg.data_root, os.path.join(cfg.data_root, split), img_transform=img_transforms) for
                split in splits]
    img_w, img_h = cfg.img_width, cfg.img_height
    row_anchor = gen_row_anchor()

    for split, dataset in zip(splits, datasets):
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

        # init video out
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        print(split[:-3] + 'avi')
        vout = cv2.VideoWriter(split[:-3] + 'avi', fourcc, 30.0, (img_w, img_h))

        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, names = data
            imgs = imgs.cuda()
            with torch.no_grad():
                out = net(imgs)

            col_sample_w = (cfg.train_img_width - 1) / (cfg.griding_num - 1)

            out_j = out[0].data.cpu().numpy()
            out_j2 = out_j
            # print(out)
            # print(out_j, flush=True)
            out_j = out_j[:, ::-1, :]
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(cfg.griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == cfg.griding_num] = 0
            out_j = loc

            print(out_j, flush=True)
            print('-------------')
            print(out_j2, flush=True)

            print('hi', flush=True)
            import pdb; pdb.set_trace()
            vis = cv2.imread(os.path.join(cfg.data_root, names[0]))
            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            ppp = (
                                int(out_j[k, i] * col_sample_w * img_w / global_config.cfg.train_img_width) - 1,
                                int(img_h * (row_anchor[cfg.cls_num_per_lane - 1 - k] /
                                             global_config.cfg.train_img_height)) - 1)
                            cv2.circle(vis, ppp, 5, (0, 255, 0), -1)
            vout.write(vis)

        vout.release()
