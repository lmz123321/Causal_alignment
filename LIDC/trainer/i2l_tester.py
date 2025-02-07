import torch
import sys
sys.path.append("/home/lijingwen/Projects")
from Counter_align.LIDC.CCCE.CCCE_test import pred_to_ccce
import Counter_align.LIDC.utils.metrics as module_metrics
import Counter_align.LIDC.data_loader.data_loaders as module_data
import Counter_align.LIDC.model.attention as module_atter
import Counter_align.LIDC.model.classifier as module_clser
from Counter_align.LIDC.utils.util import att_preds_totensor, digitize, MetricTracker, modify_att, get_topk
class WholeTester:  # whole hierachical tester
    '''
    whole hierachical tester
    '''

    def __init__(self, device, config):
        self.config = config
        self.logger = config.get_logger('trainer', 2)
        self.device = device

        # dataloader
        self.data_loader = getattr(module_data, config['data_loader']['type'])(
            cache_path=config['data_loader']['args']['cache_path'],
            ccce_file=config['data_loader']['args']['ccce_file'],
            batch_size=config['data_loader']['args']['batch_size'],
            training=False, valid=False, shuffle=True, seed=config['seed']
        )
        # i2l clser,atter
        self.i2l_clser = getattr(module_clser, config["i2l_clser"]["arch"])(in_dim=config["i2l_clser"]["in_dim"]).to(device)
        self.i2l_clser.load_state_dict(torch.load(config['i2l_clser']['checkpoint_path'], map_location=device))
        self.i2l_atter = getattr(module_atter, config["i2l_atter"]["arch"])(in_dim=config["i2l_atter"]["in_dim"]).to(device)

        self.i2l_atter.load_state_dict(torch.load(config['i2l_atter']['checkpoint_path'], map_location=device))

   
        # metrics writer should follow
        self.i2l_clser_metrics = [getattr(module_metrics, met) for met in config["i2l_clser"]["metrics"]]
        self.i2l_atter_metrics = [getattr(module_metrics, met) for met in config["i2l_atter"]["metrics"]]
    
        # metric trackers
        i2l_clser_metric_names = ['i2l_avg_acc']
        i2l_atter_metric_names = ['i2l_prec', 'i2l_rec', 'i2l_dice']
      
        self.i2l_clser_trackers = MetricTracker(*i2l_clser_metric_names,writer=None)
        self.i2l_atter_trackers = MetricTracker(*i2l_atter_metric_names,writer=None)

        # public_tensors in 4 subepochs
        self.public_tensors = {'i2l_attention_mask': []}



    def test(self):
        """
        Full training logic
        sub_testers definition
        for epoch in range(0, self.epochs):
            sub_testers trainning
        """
        # self.mask_gt[ind], self.att_gt[ind], self.ccce_gt[ind], self.z[ind], self.y[ind]
        self.i2l_atter.eval()
        self.public_tensors['i2l_attention_mask'] = []
        for batch_idx, (mask_gt, _, _, z, _) in enumerate(self.data_loader):
            z = z.to(self.device)
            mask_gt = mask_gt.to(self.device)
            with torch.no_grad():
                i2l_attention_mask = digitize(self.i2l_atter(z.to(self.device)).float().detach().clone())
                self.public_tensors['i2l_attention_mask'].append(i2l_attention_mask)
                idx = 0
                for met in self.i2l_atter_metrics:
                    self.i2l_atter_trackers.update(self.i2l_atter_trackers.keys[idx], met(i2l_attention_mask, mask_gt))
                    idx += 1
        log_i2l_atter = self.i2l_atter_trackers.result()

        for batch_idx, (_, _, _, z, y) in enumerate(self.data_loader):
            mask = self.public_tensors['i2l_attention_mask'][batch_idx].to(self.device)
            z = z.to(self.device)
            y = y.long().unsqueeze(1).to(self.device)

            with torch.no_grad():
                preds = self.i2l_clser(mask * z)

                idx = 0
                for met in self.i2l_clser_metrics:
                    self.i2l_clser_trackers.update(self.i2l_clser_trackers.keys[idx], met(preds, y))
                    idx += 1

        log_i2l_clser = self.i2l_clser_trackers.result()


        log = {**log_i2l_atter, **log_i2l_clser}
        for key, value in log.items():
            print('    {:3s}: {}'.format(str(key), value))