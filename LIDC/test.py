import argparse
import collections
import os
import torch
import time
import numpy as np
from parse_config import ConfigParser
import trainer.tester as module_tester


def main(config):
    logger = config.get_logger('train')
    logger.info('Log dir: {}'.format(config._log_dir))
    time.sleep(5)
    tester = getattr(module_tester, config['trainer']['type'])\
        (config=config,device=device)

    tester.test()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-id', '--runid', default=None, type=str,
                      help='running experiment id (default: date_time)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-gpu', '--gpu'],
                   type=str, target='gpu'),
        CustomArgs(['-bs', '--batch_size'], type=int,
                   target='data_loader;args;batch_size'),
        CustomArgs(['-seed', '--rand_seed'], type=int,
                   target='seed'),
        CustomArgs(['-if_ccce_match_att', '--if_ccce_match_att'],
                   type=int, target='trainer;args;if_ccce_match_att'),  # todo by ljw 20240701 0->false 1->true
        CustomArgs(['-a2l_att_thresh', '--a2l_atter_thresh_hold'],
                   type=float, target='a2l_atter;thresh_hold'),#todo by ljw 20240701

    ]
    config = ConfigParser.from_args(args, options, save=True)

    # fix random seeds for reproducibility
    torch.manual_seed(config['seed'])
    # to speed up
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = 'cuda'
    np.random.seed(config['seed'])

    main(config)

