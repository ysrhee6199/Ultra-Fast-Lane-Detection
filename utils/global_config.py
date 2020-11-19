import sys

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


def init():
    """
    in the good old times this was a simple import and forget, but this behaviour breaks sphinx :/
    Now the config has to be initialized once at start
    """
    global cfg, adv_cfg, args
    if not cfg:
        args, cfg = merge_config()
        adv_cfg = _abc()


class Dummy:
    """
    This is a simple dummy class (mock) which will always return a dummy string for every value its asked for
    This prevents errors if this applications is run under unusual circumstances (eg doc generation)
    """
    def __getattribute__(self, item):
        return "DUMMY_CFG_VALUE"


# Use mock class if this file is called during doc generation
if 'sphinx' in sys.modules:
    cfg = Dummy()
    args = Dummy()
    adv_cfg = Dummy()
