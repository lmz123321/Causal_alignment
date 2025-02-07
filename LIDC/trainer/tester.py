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
        self.thresh_hold =config["a2l_atter"]["thresh_hold"]
        self.if_ccce_match_att =config['trainer']['args']['if_ccce_match_att']

        # dataloader
        self.data_loader = getattr(module_data, config['data_loader']['type'])(
            cache_path=config['data_loader']['args']['cache_path'],
            ccce_file=config['data_loader']['args']['ccce_file'],
            batch_size=config['data_loader']['args']['batch_size'],
            training=False, valid=False, shuffle=True, seed=config['seed']
        )
        # i2a clser,atter
        self.i2a_clser = getattr(module_clser, config["i2a_clser"]["arch"])(in_dim=config["i2a_clser"]["in_dim"]).to(device)
        self.i2a_clser.load_state_dict(torch.load(config['i2a_clser']['checkpoint_path'], map_location=device))
        self.i2a_atter = getattr(module_atter, config["i2a_atter"]["arch"])(in_dim=config["i2a_atter"]["in_dim"]).to(device)

        self.i2a_atter.load_state_dict(torch.load(config['i2a_atter']['checkpoint_path'], map_location=device))

        # a2l clser,atter;load checkpoint pt
        self.a2l_clser = getattr(module_clser, config["a2l_clser"]["arch"])(in_dim=config["a2l_clser"]["in_dim"]).to(device)
        self.a2l_clser.load_state_dict(torch.load(config['a2l_clser']['checkpoint_path'], map_location=device))

        self.a2l_atter = getattr(module_atter, config["a2l_atter"]["arch"])(in_dim=config["a2l_atter"]["in_dim"],extend_dim=config["a2l_atter"]["extend_dim"],
                                                                            out_dim=config["a2l_atter"]["out_dim"],C=config["a2l_atter"]["C"]).to(device)
        self.a2l_atter.load_state_dict(torch.load(config['a2l_atter']['checkpoint_path'], map_location=device))


        # metrics writer should follow
        self.i2a_clser_metrics = [getattr(module_metrics, met) for met in config["i2a_clser"]["metrics"]]
        self.i2a_atter_metrics = [getattr(module_metrics, met) for met in config["i2a_atter"]["metrics"]]
        self.a2l_clser_metrics = [getattr(module_metrics, met) for met in config["a2l_clser"]["metrics"]]
        self.a2l_atter_metrics = [getattr(module_metrics, met) for met in config["a2l_atter"]["metrics"]]
        # metric trackers
        i2a_clser_metric_names = ['i2a_avg_acc']
        a2l_clser_metric_names = ['a2l_avg_acc']
        i2a_atter_metric_names = ['i2a_prec', 'i2a_rec', 'i2a_dice']
        a2l_atter_metric_names = ['ccce_acc','a2l_prec']
        self.i2a_clser_trackers = MetricTracker(*i2a_clser_metric_names,writer=None)
        self.a2l_clser_trackers = MetricTracker(*a2l_clser_metric_names,writer=None)
        self.i2a_atter_trackers = MetricTracker(*i2a_atter_metric_names,writer=None)
        self.a2l_atter_trackers = MetricTracker(*a2l_atter_metric_names,writer=None)
        # public_tensors in 4 subepochs
        self.public_tensors = {'i2a_attention_mask': [], 'a2l_attention_mask': [],'i2a_att_feat':[] ,'i2a_att_pred': [],'pred_ccce_gt':[]}

    def get_pred_match_ccce(self):
        self.public_tensors['pred_ccce_gt'] = []
        topk=int(str(self.config["data_loader"]["args"]["ccce_file"])[-5])

        with torch.no_grad():
            for batch_idx, (_, _, _, _, y) in enumerate(self.data_loader):
                y=y.to(self.device)
                pred_att = self.public_tensors['i2a_att_pred'][batch_idx].to(
                    self.device)
                pred_ccce_gt=pred_to_ccce(pred_att, y,
                             ['subtlety','calcification','margin','spiculation','lobulation','texture'],
                             'malignancy', topk).to(self.device)
                self.public_tensors['pred_ccce_gt'].append(pred_ccce_gt)

    def test(self):
        """
        Full training logic
        sub_testers definition
        for epoch in range(0, self.epochs):
            sub_testers trainning
        """
        # self.mask_gt[ind], self.att_gt[ind], self.ccce_gt[ind], self.z[ind], self.y[ind]
        self.i2a_atter.eval()
        self.public_tensors['i2a_attention_mask'] = []
        for batch_idx, (mask_gt, _, _, z, _) in enumerate(self.data_loader):
            z = z.to(self.device)
            mask_gt = mask_gt.to(self.device)
            with torch.no_grad():
                i2a_attention_mask = digitize(self.i2a_atter(z.to(self.device)).float().detach().clone())
                self.public_tensors['i2a_attention_mask'].append(i2a_attention_mask)
                idx = 0
                for met in self.i2a_atter_metrics:
                    self.i2a_atter_trackers.update(self.i2a_atter_trackers.keys[idx], met(i2a_attention_mask, mask_gt))
                    idx += 1
        log_i2a_atter = self.i2a_atter_trackers.result()

        for batch_idx, (_, att_gt, _, z, _) in enumerate(self.data_loader):
            mask = self.public_tensors['i2a_attention_mask'][batch_idx].to(self.device)
            z = z.to(self.device)
            att_gt = att_gt.long().to(self.device)

            with torch.no_grad():
                pred_feat = self.i2a_clser.forward_feature(mask * z)
                preds = self.i2a_clser(mask * z)
                i2a_att_pred = att_preds_totensor(preds).detach().clone()
                self.public_tensors['i2a_att_pred'].append(i2a_att_pred)
                self.public_tensors['i2a_att_feat'].append(pred_feat)
                idx = 0
                for met in self.i2a_clser_metrics:
                    self.i2a_clser_trackers.update(self.i2a_clser_trackers.keys[idx], met(preds, att_gt))
                    idx += 1

        log_i2a_clser = self.i2a_clser_trackers.result()

        if self.if_ccce_match_att == 1:
            self.get_pred_match_ccce()
        for batch_idx, (_, _, _, _, _) in enumerate(self.data_loader):
            ccce_gt =  self.public_tensors['pred_ccce_gt'][batch_idx]
            ccce_expanded = ccce_gt.unsqueeze(-1).repeat(1, 1, 16).to(self.device)  # by ljw 20240627 (B,6)->(B,6,1)->(B,6,C)
            att_feat = self.public_tensors['i2a_att_feat'][batch_idx].to(self.device)
            att_feat = att_feat.reshape(att_feat.size(0), -1)  # by ljw 20240625 (B,6,C)->(B,6*C)

            with torch.no_grad():
                if self.thresh_hold < 1:
                    a2l_attention_mask = digitize(
                        self.a2l_atter(att_feat.to(self.device)).float().detach().clone(),
                        self.thresh_hold)
                else:
                    a2l_attention_mask = get_topk(
                        self.a2l_atter(att_feat.to(self.device)).float().detach().clone(), C=16,
                        topk=self.thresh_hold)
                self.public_tensors['a2l_attention_mask'].append(a2l_attention_mask)
                idx = 0
                for met in self.a2l_atter_metrics:
                    self.a2l_atter_trackers.update(self.a2l_atter_trackers.keys[idx], met(a2l_attention_mask, ccce_expanded.reshape(ccce_expanded.size(0),-1)))
                    idx += 1
        log_a2l_atter = self.a2l_atter_trackers.result()


        for batch_idx, (_, _, _, _, y) in enumerate(self.data_loader):
            ccce_pred = self.public_tensors['a2l_attention_mask'][batch_idx].reshape(y.size(0), 6,-1).float().to(
                self.device)  # (B,6)->(B,6,1) or (B,96)->(B,6,16)
            y = y.to(self.device)
            att_pred = self.public_tensors['i2a_att_feat'][batch_idx].float().to( self.device)  # (B,6,C)

            with torch.no_grad():
                preds = self.a2l_clser((att_pred * ccce_pred).reshape(y.size(0), -1))
                idx = 0
                for met in self.a2l_clser_metrics:
                    self.a2l_clser_trackers.update(self.a2l_clser_trackers.keys[idx], met(preds, y.int()))
                    idx += 1
        log_a2l_clser = self.a2l_clser_trackers.result()

        log = {**log_i2a_atter, **log_i2a_clser, **log_a2l_atter, **log_a2l_clser}
        for key, value in log.items():
            print('    {:3s}: {}'.format(str(key), value))