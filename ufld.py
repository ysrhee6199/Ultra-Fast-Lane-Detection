from src import runtime, train

from src.utils.global_config import cfg

if __name__ == "__main__":
    # do some basic cfg validation and call runtime or train according to mode

    if not cfg.data_root or not cfg.work_dir:
        raise Exception('data_root and work_dir have to be specified')

    if cfg.mode == 'runtime':
        if not cfg.trained_model:
            raise Exception('define your trained_model')
        runtime.main()
    elif cfg.mode == 'train':
        train.main()
    else:
        raise Exception('invalid mode')