import yaml
from easydict import EasyDict
import os
import torch
import time
from .logger import *
from utils import dist_utils, misc

def log_args_to_file(args, pre='args', logger=None):
    for key, val in args.items():
        print_log(f'{pre}.{key} : {val}', logger = logger)

def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            print_log(f'{pre}.{key} = edict()', logger = logger)
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        print_log(f'{pre}.{key} : {val}', logger = logger)

def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config)        
    return config

def get_config(args, logger=None):
    if args['resume']:
        cfg_path = os.path.join(args['experiment_path'], 'config.yaml')
        if not os.path.exists(cfg_path):
            print_log("Failed to resume", logger = logger)
            raise FileNotFoundError()
        print_log(f'Resume yaml from {cfg_path}', logger = logger)
        args['config'] = cfg_path
    config = cfg_from_yaml_file(args['config_backbone'])
    if not args['resume'] and args['local_rank'] == 0:
        save_experiment_config(args, config, logger)
    return config

def save_experiment_config(args, config, logger = None):
    config_path = os.path.join(args['experiment_path'], 'config.yaml')
    os.system('cp %s %s' % (args['config'], config_path))
    print_log(f"Copy the Config file from {args['config']} to {config_path}",logger = logger )

def log_model_config(args):
    # log the model
    # CUDA
    args['use_gpu'] = torch.cuda.is_available()
    if args['use_gpu']:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args['launcher'] == 'none':
        args['distributed'] = False
    else:
        args['distributed'] = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args['experiment_path'], f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args['log_name'])
    # config
    config = get_config(args, logger=logger)
    # batch size
    if args['distributed']:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs // world_size * 2
        config.dataset.val.others.bs = config.total_bs // world_size * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs // world_size
    else:
        config.dataset.train.others.bs = config.total_bs
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs * 2
        config.dataset.val.others.bs = config.total_bs * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs
            # log
    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)
    # exit()
    logger.info(f"Distributed training: {args['distributed']}")
    # set random seeds
    if args['seed'] is not None:
        logger.info(f"Set random seed to {args['seed']}, "
                    f"deterministic: {args['deterministic']}")
        misc.set_random_seed(args['seed'] + args['local_rank'],
                             deterministic=args['deterministic'])  # seed + rank, for augmentation
    if args['distributed']:
        assert args['local_rank'] == torch.distributed.get_rank()

    if args['shot'] != -1:
        config.dataset.train.others.shot = args['shot']
        config.dataset.train.others.way = args['way']
        config.dataset.train.others.fold = args['fold']
        config.dataset.val.others.shot = args['shot']
        config.dataset.val.others.way = args['way']
        config.dataset.val.others.fold = args['fold']

    return  args, config, logger
    ###############