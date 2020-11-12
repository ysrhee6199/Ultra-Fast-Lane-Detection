from utils.common import merge_config

cfg = None
args = None

if not cfg:
    args, cfg = merge_config()
