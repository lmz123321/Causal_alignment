import argparse
import collections
import os
import torch
import time
import numpy as np
from parse_config import ConfigParser
import trainer.trainer as module_trainer


def main(config):
    logger = config.get_logger('train')
    logger.info('Log dir: {}'.format(config._log_dir))
    time.sleep(5)
    trainer =  getattr(module_trainer, config['trainer']['type'])\
        (config=config,device=device)

    trainer.train()
    trainer.save_code()

    # trainer.train_load()


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
        CustomArgs(['-epochs', '--epochs'],
                   type=int, target='trainer;args;epochs'),
        CustomArgs(['-if_ccce_match_att', '--if_ccce_match_att'],
                   type=int, target='trainer;args;if_ccce_match_att'),  # todo by ljw 20240701 0->false 1->true


        CustomArgs(['-i2a_cls_lr', '--i2a_clser_learning_rate'],
                   type=float, target='i2a_clser;lr'),
        CustomArgs(['-i2a_cls_sub', '--i2a_clser_subepochs'],
                   type=int, target='i2a_clser;subepochs'),

        CustomArgs(['-i2a_att_lr', '--i2a_atter_learning_rate'],
                   type=float, target='i2a_atter;lr'),
        CustomArgs(['-i2a_att_sub', '--i2a_atter_subepochs'],
                   type=int, target='i2a_atter;subepochs'),
        CustomArgs(['-i2a_att_lamb', '--i2a_atter_lamb'],
                   type=float, target='i2a_atter;lamb'),
        CustomArgs(['-i2a_att_mult', '--i2a_atter_mult'],
                   type=float, target='i2a_atter;mult'),

        CustomArgs(['-a2l_cls_lr', '--a2l_clser_learning_rate'],
                   type=float, target='a2l_clser;lr'),
        CustomArgs(['-a2l_cls_sub', '--a2l_clser_subepochs'],
                   type=int, target='a2l_clser;subepochs'),

        CustomArgs(['-a2l_att_lr', '--a2l_atter_learning_rate'],
                   type=float, target='a2l_atter;lr'),
        CustomArgs(['-a2l_att_sub', '--a2l_atter_subepochs'],
                   type=int, target='a2l_atter;subepochs'),
        CustomArgs(['-a2l_att_lamb', '--a2l_atter_lamb'],
                   type=float, target='a2l_atter;lamb'),
        CustomArgs(['-a2l_att_ratio', '--a2l_atter_ratio'],
                   type=float, target='a2l_atter;ratio'),
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

