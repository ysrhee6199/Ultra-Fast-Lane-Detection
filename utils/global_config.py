from torchvision import transforms

from utils.common import merge_config

cfg = None
args = None
adv_cfg = None


class _abc:
    """
    This class provides "advanced config values" meaning it calculates some values which depend on other cfg values
    They are calculated here to
    - keep the configs clean
    - prevent calculating them multiple times during runtime which should make your code cleaner and improve performance
    """

    def __init__(self):
        self.cls_num_per_lane = len(cfg.h_samples)
        self.scaled_h_samples = [int(round(x * cfg.img_height)) for x in cfg.h_samples]
        self.train_h_samples = [x * cfg.train_img_height for x in cfg.h_samples]
        self.img_transform = transforms.Compose([
            transforms.Resize((cfg.train_img_height, cfg.train_img_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


# init
if not cfg:
    args, cfg = merge_config()
    adv_cfg = _abc()
