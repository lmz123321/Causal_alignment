import argparse
import collections
import torch
import time
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model as module_arch
from parse_config import ConfigParser
import trainer as module_trainer
from utils import prepare_device


def main(config):
    logger = config.get_logger('train')
    logger.info('Log dir: {}'.format(config._log_dir))
    time.sleep(5)

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        img_size = config['data_loader']['args']['img_size'],
        shuffle=False,
        validation_split=0.0,
        training=False,
        valid=True,
        num_workers=16
    )

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    trainer = getattr(module_trainer,config['trainer']['type'])(
        model, criterion, metrics, optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader
    )
    trainer.train()
    trainer.save_code()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-id', '--runid', default=None, type=str,
                      help='running experiment id (default: date_time)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-lr', '--learning_rate'],
                   type=float, target='optimizer;args;lr'),
        CustomArgs(['-bs', '--batch_size'], type=int,
                   target='data_loader;args;batch_size'),
        CustomArgs(['-seed', '--rand_seed'], type=int,
                   target='seed')
    ]
    config = ConfigParser.from_args(args, options, save=True)

    # fix random seeds for reproducibility
    SEED = config['seed']
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    main(config)
