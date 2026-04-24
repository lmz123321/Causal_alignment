import argparse
import torch
import collections
import os
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model as module_arch
from parse_config import ConfigParser
from torch.utils.tensorboard import SummaryWriter
from utils import nesteddict2dict,figure_to_image
import matplotlib.pyplot as plt

def main(config):
    logger = config.get_logger('test')
    # build model architecture
    model = config.init_obj('arch', module_arch)
    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    config.resume = os.path.join(config['trainer']['args']['save_dir'],config.run_id,'models/model_best.pth')
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    # prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    logs = dict()
    visualizations = list()
    for subset in ['val','test']:
        # setup data_loader instances
        data_loader = getattr(module_data, config['data_loader']['type'])(
            config['data_loader']['args']['data_dir'],
            batch_size=256,
            img_size=config['data_loader']['args']['img_size'],
            shuffle=False,
            validation_split=0.0,
            training=(subset=='train'),
            valid=(subset=='val'),
            num_workers=16)

        total_loss = 0.0
        total_metrics = torch.zeros(len(metric_fns))
        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(data_loader,desc=subset)):
                data, target = data.to(device), target.to(device)
                output = model(data)
                # save visualization
                if subset=='test' and 'autoencoder' in config['name']:
                    visualizations.append((data.detach().cpu(),output.detach().cpu()))
                # computing loss, metrics on test set
                loss = loss_fn(output, data.detach()) if 'autoencoder' in config['name'] else loss_fn(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(metric_fns):
                    total_metrics[i] += metric(output, target) * batch_size

        n_samples = len(data_loader.sampler)
        log = {'{}/loss'.format(subset): total_loss / n_samples}
        log.update({'{}/{}'.format(subset,met.__name__):
                        total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
        logs.update(log)
        logger.info(log)

    # hyper-params & metric monitor with tensorboard
    to_monitor = config['to_monitor']
    dic = nesteddict2dict(config.config)
    hparams = dict()
    for key in dic.keys():
        if key in to_monitor.keys():
            hparams[to_monitor[key]] = dic[key]
    with SummaryWriter(log_dir=config.log_dir) as w:
        w.add_hparams(hparam_dict=hparams,metric_dict=logs,run_name=config.run_id)
        if 'autoencoder' in config['name']:
            num_show = config['tensorboard']['num_show']; plot_size = config['tensorboard']['plot_size']
            origs,recs = zip(*visualizations)
            origs, recs = torch.cat(origs,dim=0), torch.cat(recs,dim=0)
            fig, ax = plt.subplots(num_show, 2, figsize=(2 * plot_size, num_show * plot_size))
            for ind in range(num_show):
                ax[ind][0].imshow(origs[ind, 0], cmap='gray')
                ax[ind][1].imshow(recs[ind, 0], cmap='gray')
                ax[ind][0].text(0, -10, data_loader.dataset.img_list[ind].split('/')[-1][:-4], size=5*plot_size)
                ax[ind][0].text(0, 0, 'orig', size=5*plot_size)
                ax[ind][1].text(0, 0, 'rec', size=5*plot_size)
            for _ax in ax.flatten():
                _ax.axis('off')
            w.add_image(config.run_id, figure_to_image(fig))


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
                   target='seed'),
        CustomArgs(['-init', '--init_features'], type=int,
                   target='arch;args;init_features'),
    ]
    config = ConfigParser.from_args(args, options, save=True)
    main(config)
