from utils.common import merge_config


def init():
    global cfg
    global args

    args, cfg = merge_config()
