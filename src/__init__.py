# init config, this has to be done first as its values are used in method declarations
from .utils import global_config

global_config.init()

from .runtime import runtime
from .train import train

__all__ = ['runtime', 'train']
