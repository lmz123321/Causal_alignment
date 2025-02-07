import os
from copy import deepcopy

import torch
from abc import abstractmethod

import torchopt
from numpy import inf
import sys
sys.path.append("/home/lijingwen/Projects")
from Counter_align.DDSM.CCCE.CCCE_test import pred_to_ccce
import Counter_align.DDSM.loss.loss as module_losses
import Counter_align.DDSM.utils.metrics as module_metrics
import Counter_align.DDSM.data_loader.data_loaders as module_data
from Counter_align.DDSM.implicit_gradient.attributes2label import get_implicit_a2lmodel
from Counter_align.DDSM.implicit_gradient.image2attributes import get_implicit_i2amodel
import Counter_align.DDSM.model.attention as module_atter
import Counter_align.DDSM.model.classifier as module_clser
from Counter_align.DDSM.utils.util import att_preds_totensor, digitize, MetricTracker, modify_att, get_topk
from logger import TensorboardWriter
import shutil
from tqdm import tqdm


class WholeTrainer:  # whole hierachical trainer
    '''
    whole hierachical trainer
    '''

    def __init__(self, device, config):
        self.config = config
        self.device=device
        cfg_trainer = config['trainer']['args']
        self.logger = config.get_logger('trainer', 2)
        self.epochs = cfg_trainer['epochs']
        self.codepath = cfg_trainer['code_source']

        # dataloader
        self.train_data_loader = getattr(module_data, config['data_loader']['type'])(
            cache_path=config['data_loader']['args']['cache_path'],
            ccce_file=config['data_loader']['args']['ccce_file'],
            batch_size=config['data_loader']['args']['batch_size'],
            training=True, valid=False, shuffle=True, seed=config['seed']
        )
        self.val_data_loader = getattr(module_data, config['data_loader']['type'])(
            cache_path=config['data_loader']['args']['cache_path'],
            ccce_file=config['data_loader']['args']['ccce_file'],
            training=False, valid=True, shuffle=False, seed=config['seed']
        )
        self.logger.info('Each train/val epoch contains {}/{} steps in tensorboard, respectively'.format(
            len(self.train_data_loader), len(self.val_data_loader)))

        # a2l clser,atter,optimizer,solver
        self.a2l_clser = getattr(module_clser, config["a2l_clser"]["arch"])(in_dim=config["a2l_clser"]["in_dim"]).to(device)
        self.a2l_atter = getattr(module_atter, config["a2l_atter"]["arch"])(in_dim=config["a2l_atter"]["in_dim"],extend_dim=config["a2l_atter"]["extend_dim"],
                                                                            out_dim=config["a2l_atter"]["out_dim"],C=config["a2l_atter"]["C"]).to(device)

        self.optim_a2l_clser = torch.optim.Adam(self.a2l_clser.parameters(), lr=config["a2l_clser"]["lr"])
        self.optim_a2l_atter = torch.optim.Adam(self.a2l_atter.parameters(), lr=config["a2l_atter"]["lr"])

        self.a2l_solver = torchopt.linear_solve.solve_normal_cg(rtol=config["a2l_atter"]["rtol"],
                                                                maxiter=config["a2l_atter"]["maxiter"])
        self.a2l_imp = get_implicit_a2lmodel(self.a2l_solver)(self.a2l_clser, self.a2l_atter, 0.1)

        # i2a clser,atter,optimizer,solver
        self.i2a_clser = getattr(module_clser, config["i2a_clser"]["arch"])(in_dim=config["i2a_clser"]["in_dim"]).to(device)
        self.i2a_atter = getattr(module_atter, config["i2a_atter"]["arch"])(in_dim=config["i2a_atter"]["in_dim"]).to(device)

        self.optim_i2a_clser = torch.optim.Adam(self.i2a_clser.parameters(), lr=config["i2a_clser"]["lr"])
        self.optim_i2a_atter = torch.optim.Adam(self.i2a_atter.parameters(), lr=config["i2a_atter"]["lr"])
        self.i2a_solver = torchopt.linear_solve.solve_normal_cg(rtol=config["i2a_atter"]["rtol"],
                                                                maxiter=config["i2a_atter"]["maxiter"])
        self.i2a_imp = get_implicit_i2amodel(self.i2a_solver)(self.i2a_clser, self.i2a_atter, 0.1)

        # metrics writer should follow
        self.i2a_clser_metrics = [getattr(module_metrics, met) for met in config["i2a_clser"]["metrics"]]
        self.i2a_atter_metrics = [getattr(module_metrics, met) for met in config["i2a_atter"]["metrics"]]
        self.a2l_clser_metrics = [getattr(module_metrics, met) for met in config["a2l_clser"]["metrics"]]
        self.a2l_atter_metrics = [getattr(module_metrics, met) for met in config["a2l_atter"]["metrics"]]

        # losses writer should follow
        self.i2a_clser_losses = [getattr(module_losses,loss) for loss in config["i2a_clser"]["losses"]]
        self.i2a_atter_losses = [getattr(module_losses,loss) for loss in config["i2a_atter"]["losses"]]
        self.a2l_clser_losses = [getattr(module_losses,loss) for loss in config["a2l_clser"]["losses"]]
        self.a2l_atter_losses = [getattr(module_losses,loss) for loss in config["a2l_atter"]["losses"]]

        # loss should monitor,save best chekpoint in this epoch based on these loss performance
        self.i2a_clser_loss = getattr(module_losses, config["i2a_clser"]["loss"])
        self.i2a_atter_loss = getattr(module_losses, config["i2a_atter"]["loss"])
        self.a2l_clser_loss = getattr(module_losses, config["a2l_clser"]["loss"])
        self.a2l_atter_loss = getattr(module_losses, config["a2l_atter"]["loss"])

        # public_tensors in 4 subepochs
        self.public_tensors = {'train_i2a_attention_mask': [mask_gt for mask_gt, _, _, _, _ in self.train_data_loader],
                               'val_i2a_attention_mask': [mask_gt for mask_gt, _, _, _, _ in self.val_data_loader],
                               'train_a2l_attention_mask': [ccce_gt for _, _, ccce_gt,_, _ in self.train_data_loader],
                               'val_a2l_attention_mask': [ccce_gt for _, _, ccce_gt, _, _ in self.val_data_loader],
                               'train_i2a_att_feat': [], 'val_i2a_att_feat': [],
                               'train_modified_att': [], 'val_modified_att': [],
                               'i2a_clser': [], 'i2a_atter': [], 'a2l_clser': [], 'a2l_atter': [],
                               'train_pred_ccce_gt':[ccce_gt for _, _, ccce_gt,_, _ in self.train_data_loader],  #todo by ljw 20240702
                               'val_pred_ccce_gt':[ccce_gt for _, _, ccce_gt,_, _ in self.val_data_loader], #todo by ljw 20240702
                               'train_i2a_att_pred': [], # todo by ljw 20240702
                               'val_i2a_att_pred': []# todo by ljw 20240702
                               }

        self.code_dir = config.code_dir
        # all the sub_trainer share writer
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])


    def log_hyperparams_as_table(self):
        config_set = [
            ['Parameter', 'Value'],
            ['gpu', self.config['gpu']],
            ['seed', self.config['seed']],
            ['topk', str(self.config["data_loader"]["args"]["ccce_file"])[-5]],
            ['epochs', self.config['trainer']['args']['epochs']],
            ['i2a_cls_lr', self.config['i2a_clser']['lr']],
            ['i2a_cls_sub', self.config['i2a_clser']['subepochs']],
            ['i2a_att_lr', self.config['i2a_atter']['lr']],
            ['i2a_att_sub', self.config['i2a_atter']['subepochs']],
            ['i2a_att_lamb', self.config['i2a_atter']['lamb']],
            ['i2a_att_mult', self.config['i2a_atter']['mult']],#by ljw 20240731
            ['a2l_cls_lr', self.config['a2l_clser']['lr']],
            ['a2l_cls_sub', self.config['a2l_clser']['subepochs']],
            ['a2l_att_lr', self.config['a2l_atter']['lr']],
            ['a2l_att_sub', self.config['a2l_atter']['subepochs']],
            ['a2l_att_lamb', self.config['a2l_atter']['lamb']],
            ['a2l_att_ratio', self.config['a2l_atter']['ratio']],
            ['a2l_att_thresh', self.config['a2l_atter']['thresh_hold']],# todo by ljw 20240701
            ['if_ccce_match_att', self.config['trainer']['args']['if_ccce_match_att']],  # todo by ljw 20240701

        ]
        self.writer.add_table('Hyperparameters', config_set)

    def train(self):
        """
        Full training logic
        sub_trainers definition
        for epoch in range(0, self.epochs):
            sub_trainers trainning
        """
        self.log_hyperparams_as_table()
        # sub_trainers definition
        self.i2a_clser_trainer = I2A_Clser_Trainer(self.i2a_clser, self.optim_i2a_clser, self.i2a_clser_metrics,
                                                   self.i2a_clser_losses, self.i2a_clser_loss, self)
        self.a2l_clser_trainer = A2L_Clser_Trainer(self.a2l_clser, self.optim_a2l_clser, self.a2l_clser_metrics,
                                                   self.a2l_clser_losses, self.a2l_clser_loss, self)
        self.i2a_atter_trainer = I2A_Atter_Trainer(self.i2a_atter, self.i2a_imp, self.optim_i2a_atter,
                                                   self.i2a_atter_metrics,self.i2a_atter_losses, self.i2a_atter_loss, self)
        self.a2l_atter_trainer = A2L_Atter_Trainer(self.a2l_atter, self.a2l_imp, self.optim_a2l_atter,
                                                   self.a2l_atter_metrics,self.a2l_atter_losses, self.a2l_atter_loss, self)

        self.save_code()
        for epoch in range(0, self.epochs):
            # print epoch information to the screen
            self.logger.info('    {:3s}: {}'.format('epoch', epoch))
            self.i2a_clser_trainer.train(epoch)
            self.a2l_clser_trainer.train(epoch)#att_pred
            self.a2l_atter_trainer.train(epoch)
            self.i2a_atter_trainer.train(epoch)


    def save_code(self):
        shutil.copytree(str(self.codepath), str(self.code_dir), dirs_exist_ok=True)


class I2A_Clser_Trainer:
    '''
    Base class for trainers with alignment loss
    '''

    def __init__(self, i2a_clser, optim_i2a_clser, i2a_clser_metrics, i2a_clser_losses, i2a_clser_loss, whole_trainer):
        self.i2a_clser = i2a_clser
        self.i2a_clser_loss = i2a_clser_loss
        self.i2a_clser_metrics = i2a_clser_metrics
        self.i2a_clser_losses = i2a_clser_losses
        self.optim_i2a_clser = optim_i2a_clser
        self.i2a_clser_subepochs = whole_trainer.config["i2a_clser"]["subepochs"]
        self.monitor = whole_trainer.config["i2a_clser"]["monitor"]


        self.mnt_mode, self.mnt_metric = self.monitor.split()
        assert self.mnt_mode in ['min', 'max']
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf

        self.checkpoint_dir = os.path.join(whole_trainer.config.save_dir, 'i2a_clser')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.cls_tosave = dict()

        self.whole_trainer = whole_trainer  # whole_trainer will bring public data_loader,logger,device,writer and publis_tensors
        self.do_valation = self.whole_trainer.val_data_loader is not None

        # metris writer should follow
        metric_names = ['i2a_avg_acc']
        self.train_metrics = MetricTracker(*metric_names, writer=self.whole_trainer.writer)
        self.val_metrics = MetricTracker(*metric_names,writer=self.whole_trainer.writer)
        # losses writer should follow
        loss_names = ['i2a_celoss']
        self.train_losses = MetricTracker(*loss_names, writer=self.whole_trainer.writer)
        self.val_losses = MetricTracker(*loss_names, writer=self.whole_trainer.writer)

    def _train_subepoch(self, epoch, subepoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        self.i2a_clser.train()
        self.train_metrics.reset()
        self.train_losses.reset()
        for batch_idx, (_, att_gt, _, z, _) in enumerate(self.whole_trainer.train_data_loader):
            train_mask = self.whole_trainer.public_tensors['train_i2a_attention_mask'][batch_idx].to(
                self.whole_trainer.device)
            train_z = z.to(self.whole_trainer.device)
            train_att = att_gt.long().to(self.whole_trainer.device)
            self.optim_i2a_clser.zero_grad()
            preds = self.i2a_clser(train_mask * train_z)
            celoss = self.i2a_clser_loss(preds, train_att)
            celoss.backward()
            self.optim_i2a_clser.step()

            global_step=(epoch * self.i2a_clser_subepochs + subepoch)*len(self.whole_trainer.train_data_loader) + batch_idx
            self.whole_trainer.writer.set_step(global_step)

            self.train_losses.update('i2a_celoss', celoss.item())
            idx=0
            for met in self.i2a_clser_metrics:
                self.train_metrics.update(self.train_metrics.keys[idx], met(preds, train_att))
                idx+=1



        log_metrics = self.train_metrics.result()
        log_losses = self.train_losses.result()
        log = {**log_metrics, **log_losses}  # merge 2 dicts
        if self.do_valation:
            val_log = self._val_subepoch(epoch, subepoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        return log

    def _val_subepoch(self, epoch, subepoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        self.i2a_clser.eval()
        self.val_metrics.reset()
        self.val_losses.reset()
        # self.mask_gt[ind],self.att_gt[ind],self.ccce_gt[ind],self.z[ind],self.y[ind]
        for batch_idx, (_, att_gt, _, z, _) in enumerate(self.whole_trainer.val_data_loader):
            val_mask = self.whole_trainer.public_tensors['val_i2a_attention_mask'][batch_idx].to(
                self.whole_trainer.device)
            val_z = z.to(self.whole_trainer.device)
            val_att = att_gt.long().to(self.whole_trainer.device)
            with torch.no_grad():
                preds = self.i2a_clser(val_mask * val_z)
                celoss = self.i2a_clser_loss(preds, val_att)
            global_step = (epoch * self.i2a_clser_subepochs + subepoch) * len(self.whole_trainer.val_data_loader) + batch_idx
            self.whole_trainer.writer.set_step(global_step,'val')
            self.val_losses.update('i2a_celoss', celoss.item())
            idx = 0
            for met in self.i2a_clser_metrics:
                self.val_metrics.update(self.val_metrics.keys[idx], met(preds, val_att))
                idx += 1
        log_metrics = self.val_metrics.result()
        log_losses = self.val_losses.result()
        return {**log_metrics, **log_losses}  # merge 2 dicts

    def train(self, epoch):
        """
        Full training logic
        for subepoch in range(0, self.i2a_clser_subepochs):
            train subepoch,val_subepoch
            evaluate model performance according to configured metric,updpate best subepoch
            save every subepoch checkpoint and update the best subepoch checkpoint
            ues best subepoch clser in this epoch,record i2a_att_pred for a2l_clser
        load best subepoch clser in this epoch, for i2a_atter imp_solver and i2a_att_pred record

        """
        self.cls_tosave = dict()
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf  # get best subepoch chkeckppoint in this epoch
        for subepoch in tqdm(range(0, self.i2a_clser_subepochs),
                             desc='gpu:{} Epoch[i2a_cls]: {}'.format(self.whole_trainer.config['gpu'], epoch)):
            result = self._train_subepoch(epoch, subepoch)
            # save logged informations into log dict
            log = {'i2a_clser_subepochs': subepoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.whole_trainer.logger.info('    {:3s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.whole_trainer.logger.warning("Warning: Metric '{}' is not found. "
                                                      "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    self.best_log = log
                    best = True

            # save all subepochs checkpoint and the best subepoch checkpoint
            self.cls_tosave['subepoch:{}'.format(subepoch)] = deepcopy(self.i2a_clser.state_dict())
            if subepoch % 20 == 0:
                torch.save(self.cls_tosave, os.path.join(self.checkpoint_dir, 'i2a_cls_epoch:{}.pt'.format(epoch)))
            if best:
                self.whole_trainer.public_tensors[
                    'i2a_clser'] = self.i2a_clser.state_dict()  # save best pt in this epoch
                torch.save(self.whole_trainer.public_tensors['i2a_clser'],
                           os.path.join(self.checkpoint_dir, 'i2a_cls_epoch:{}_best.pt'.format(epoch)))

                with torch.no_grad():  # use best clser for now,update public_tensors['train_i2a_att_feat'] public_tensors['val_i2a_att_feat']
                    self.whole_trainer.public_tensors['val_i2a_att_feat'] = []
                    self.whole_trainer.public_tensors['train_i2a_att_feat'] = []
                    self.whole_trainer.public_tensors['val_i2a_att_pred'] = []#todo by ljw 20240702
                    self.whole_trainer.public_tensors['train_i2a_att_pred'] = []#todo by ljw 20240702

                    for batch_idx, (_, _, _, z, _) in enumerate(self.whole_trainer.val_data_loader):
                        val_mask = self.whole_trainer.public_tensors['val_i2a_attention_mask'][batch_idx].to(
                            self.whole_trainer.device)
                        val_z = z.to(self.whole_trainer.device)
                        with torch.no_grad():
                            pred_feat = self.i2a_clser.forward_feature(val_mask * val_z)  # by ljw 20240625,pred_feat (B,6,C)
                            self.whole_trainer.public_tensors['val_i2a_att_feat'].append(
                                pred_feat)  #by ljw 20240625,pred change to feature format

                            preds = self.i2a_clser(val_mask * val_z)#todo by ljw 20240702
                            i2a_att_pred = att_preds_totensor(preds).detach().clone()#todo by ljw 20240702
                            self.whole_trainer.public_tensors['val_i2a_att_pred'].append(i2a_att_pred)#todo by ljw 20240702



                    for batch_idx, (_, _, _, z, _) in enumerate(self.whole_trainer.train_data_loader):
                        train_mask = self.whole_trainer.public_tensors['train_i2a_attention_mask'][batch_idx].to(
                            self.whole_trainer.device)
                        train_z = z.to(self.whole_trainer.device)
                        with torch.no_grad():
                            pred_feat=self.i2a_clser.forward_feature(train_mask * train_z)#by ljw 20240625,pred_feat (B,6,C)
                            self.whole_trainer.public_tensors['train_i2a_att_feat'].append(pred_feat)#by ljw 20240625,pred change to feature format

                            preds = self.i2a_clser(train_mask * train_z)#todo by ljw 20240702
                            i2a_att_pred = att_preds_totensor(preds).detach().clone()#todo by ljw 20240702
                            self.whole_trainer.public_tensors['train_i2a_att_pred'].append(i2a_att_pred)#todo by ljw 20240702


        self.i2a_clser.load_state_dict(self.whole_trainer.public_tensors[
                                           'i2a_clser'])  # load best pt for i2a_atter imp_solver and i2a_att_pred record


class A2L_Clser_Trainer:
    '''
    Base class for trainers with alignment loss
    '''

    def __init__(self, a2l_clser, optim_a2l_clser, a2l_clser_metrics, a2l_clser_losses, a2l_clser_loss, whole_trainer):
        self.a2l_clser = a2l_clser
        self.a2l_clser_loss = a2l_clser_loss
        self.a2l_clser_metrics = a2l_clser_metrics
        self.a2l_clser_losses = a2l_clser_losses
        self.optim_a2l_clser = optim_a2l_clser
        self.a2l_clser_subepochs = whole_trainer.config["a2l_clser"]["subepochs"]
        self.monitor = whole_trainer.config["a2l_clser"]["monitor"]

        self.mnt_mode, self.mnt_metric = self.monitor.split()
        assert self.mnt_mode in ['min', 'max']
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf

        self.checkpoint_dir = os.path.join(whole_trainer.config.save_dir, 'a2l_clser')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.cls_tosave = dict()

        self.whole_trainer = whole_trainer  # whole_trainer will bring public data_loader,logger,device,writer and publis_tensors
        self.do_valation = self.whole_trainer.val_data_loader is not None

        # metris writer should follow
        metric_names = ['a2l_avg_acc']
        self.train_metrics = MetricTracker(*metric_names, writer=self.whole_trainer.writer)
        self.val_metrics = MetricTracker(*metric_names,writer=self.whole_trainer.writer)
        # losses writer should follow
        loss_names = ['a2l_celoss']
        self.train_losses = MetricTracker(*loss_names, writer=self.whole_trainer.writer)
        self.val_losses = MetricTracker(*loss_names, writer=self.whole_trainer.writer)

    def _train_subepoch(self, epoch, subepoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        self.a2l_clser.train()
        self.train_metrics.reset()
        self.train_losses.reset()
        # self.mask_gt[ind],self.att_gt[ind],self.ccce_gt[ind],self.z[ind],self.y[ind]
        for batch_idx, (_, _, ccce_gt, _, y) in enumerate(self.whole_trainer.train_data_loader):
            train_ccce = self.whole_trainer.public_tensors['train_a2l_attention_mask'][batch_idx].reshape(y.size(0),6,-1).float().to(
                    self.whole_trainer.device)#by ljw 20240628 (B,6)->(B,6,1) or (B,96)->(B,6,16)
            train_att = self.whole_trainer.public_tensors['train_i2a_att_feat'][batch_idx].float().to(self.whole_trainer.device)#(B,6,C)
            train_y = y.to(self.whole_trainer.device)
            self.optim_a2l_clser.zero_grad()
            preds = self.a2l_clser((train_att * train_ccce).reshape(train_y.size(0), -1))# by ljw 20240625 (B,6)*(B,6)->(B,2)
            celoss = self.a2l_clser_loss(preds, train_y)
            celoss.backward()
            self.optim_a2l_clser.step()

            global_step = (epoch * self.a2l_clser_subepochs + subepoch) * len(
                self.whole_trainer.train_data_loader) + batch_idx
            self.whole_trainer.writer.set_step(global_step)

            self.train_losses.update('a2l_celoss', celoss.item())
            idx = 0
            for met in self.a2l_clser_metrics:
                self.train_metrics.update(self.train_metrics.keys[idx], met(preds, train_y.int()))
                idx+=1

        log_metrics = self.train_metrics.result()
        log_losses = self.train_losses.result()
        log = {**log_metrics, **log_losses}  # merge 2 dicts
        if self.do_valation:
            val_log = self._val_subepoch(epoch, subepoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
        return log

    def _val_subepoch(self, epoch, subepoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        self.a2l_clser.eval()
        self.val_metrics.reset()
        self.val_losses.reset()

        for batch_idx, (_, _, _, _, y) in enumerate(self.whole_trainer.val_data_loader):
            val_ccce = self.whole_trainer.public_tensors['val_a2l_attention_mask'][batch_idx].\
                reshape(y.size(0), 6,-1).float().to(self.whole_trainer.device)  # by ljw 20240628 (B,6)->(B,6,1) or (B,96)->(B,6,16)

            val_att = self.whole_trainer.public_tensors['val_i2a_att_feat'][batch_idx].float().to(self.whole_trainer.device)#(B,6,C)
            val_y = y.to(self.whole_trainer.device)

            with torch.no_grad():
                preds = self.a2l_clser((val_att * val_ccce).reshape(val_y.size(0),-1))  # by ljw 20240625 (B,6,C)*(B,6)->(B,6*C)
                celoss = self.a2l_clser_loss(preds, val_y)

            global_step = (epoch * self.a2l_clser_subepochs + subepoch) * len(
                self.whole_trainer.val_data_loader) + batch_idx
            self.whole_trainer.writer.set_step(global_step,'val')

            self.val_losses.update('a2l_celoss', celoss.item())
            idx=0
            for met in self.a2l_clser_metrics:
                self.val_metrics.update(self.val_metrics.keys[idx], met(preds, val_y.int()))
                idx+=1
        log_metrics = self.val_metrics.result()
        log_losses = self.val_losses.result()
        return {**log_metrics, **log_losses}  # merge 2 dicts

    def train(self, epoch):
        """
        Full training logic
        for subepoch in range(0, self.a2l_clser_subepochs):
            train subepoch,val_subepoch
            evaluate model performance according to configured metric,updpate best subepoch
            save every subepoch checkpoint and update the best subepoch checkpoint
        load best subepoch clser in this epoch, for a2l_atter imp_solver

        """
        self.cls_tosave = dict()
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf  # get best subepoch chkeckppoint in this epoch
        for subepoch in tqdm(range(0, self.a2l_clser_subepochs),
                             desc='gpu:{} Epoch[a2l_cls]: {}'.format(self.whole_trainer.config['gpu'], epoch)):
            result = self._train_subepoch(epoch, subepoch)
            # save logged informations into log dict
            log = {'a2l_clser_subepochs': subepoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.whole_trainer.logger.info('    {:3s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.whole_trainer.logger.warning("Warning: Metric '{}' is not found. "
                                                      "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    self.best_log = log
                    best = True

            # save all subepochs checkpoint and the best subepoch checkpoint
            self.cls_tosave['subepoch:{}'.format(subepoch)] = deepcopy(self.a2l_clser.state_dict())
            if subepoch % 20 == 0:
                torch.save(self.cls_tosave, os.path.join(self.checkpoint_dir, 'a2l_cls_epoch:{}.pt'.format(epoch)))
            if best:
                self.whole_trainer.public_tensors[
                    'a2l_clser'] = self.a2l_clser.state_dict()  # save best pt in this epoch
                torch.save(self.whole_trainer.public_tensors['a2l_clser'],
                           os.path.join(self.checkpoint_dir, 'a2l_cls_epoch:{}_best.pt'.format(epoch)))

        self.a2l_clser.load_state_dict(
            self.whole_trainer.public_tensors['a2l_clser'])  # load best pt for a2l_atter imp_solver


class A2L_Atter_Trainer:
    def __init__(self, a2l_atter, a2l_imp, optim_a2l_atter, a2l_atter_metrics, a2l_atter_losses, a2l_atter_loss,
                 whole_trainer):
        self.a2l_atter = a2l_atter
        self.a2l_imp = a2l_imp
        self.a2l_atter_loss = a2l_atter_loss
        self.a2l_atter_metrics = a2l_atter_metrics
        self.a2l_atter_losses = a2l_atter_losses
        self.optim_a2l_atter = optim_a2l_atter
        self.a2l_atter_subepochs = whole_trainer.config["a2l_atter"]["subepochs"]
        self.monitor = whole_trainer.config["a2l_atter"]["monitor"]
        self.a2l_atter_loss_ratio=whole_trainer.config["a2l_atter"]["ratio"]
        self.thresh_hold=whole_trainer.config["a2l_atter"]["thresh_hold"]#todo by ljw 20240701
        self.if_ccce_match_att = whole_trainer.config['trainer']['args']['if_ccce_match_att']  # todo by ljw 20240701

        self.tmp_modified_att = {'train_modified_att': [], 'val_modified_att':[]}

        self.mnt_mode, self.mnt_metric = self.monitor.split()
        assert self.mnt_mode in ['min', 'max']
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf

        self.checkpoint_dir = os.path.join(whole_trainer.config.save_dir, 'a2l_atter')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.att_tosave = dict()

        self.whole_trainer = whole_trainer  # whole_trainer will bring public data_loader,logger,device,writer and publis_tensors
        self.do_valation = self.whole_trainer.val_data_loader is not None

        # metris writer should follow
        metric_names = ['ccce_acc']
        self.train_metrics = MetricTracker(*metric_names, writer=self.whole_trainer.writer)
        self.val_metrics = MetricTracker(*metric_names, writer=self.whole_trainer.writer)
        # losses writer should follow
        loss_names = ['a2l_reward', 'a2l_punish', 'a2l_align_ratio', 'a2l_alignloss']
        self.train_losses = MetricTracker(*loss_names, writer=self.whole_trainer.writer)
        self.val_losses = MetricTracker(*loss_names, writer=self.whole_trainer.writer)

    def get_pred_match_ccce(self):#todo by ljw 20240701
        self.whole_trainer.public_tensors['train_pred_ccce_gt']=[]
        self.whole_trainer.public_tensors['val_pred_ccce_gt']=[]
        topk=int(str(self.whole_trainer.config["data_loader"]["args"]["ccce_file"])[-5])

        with torch.no_grad():
            for batch_idx, (_, _, _, _, y) in enumerate(self.whole_trainer.val_data_loader):
                val_y=y.to(self.whole_trainer.device)
                pred_att = self.whole_trainer.public_tensors['val_i2a_att_pred'][batch_idx].to(
                    self.whole_trainer.device)
                pred_ccce_gt=pred_to_ccce(pred_att, val_y,
                             ['subtlety','mass shape','CIRCUMSCRIBED','OBSCURED','ILL_DEFINED','SPICULATED'],
                             'pathology', topk).to(self.whole_trainer.device)

                self.whole_trainer.public_tensors['val_pred_ccce_gt'].append(pred_ccce_gt)
                # print("val_pred_ccce_gt:")
                # print(pred_ccce_gt)

            for batch_idx, (_, _, _,_, y) in enumerate(self.whole_trainer.train_data_loader):
                train_y=y.to(self.whole_trainer.device)
                pred_att = self.whole_trainer.public_tensors['train_i2a_att_pred'][batch_idx].to(
                    self.whole_trainer.device)
                pred_ccce_gt=pred_to_ccce(pred_att, train_y,
                             ['subtlety','mass shape','CIRCUMSCRIBED','OBSCURED','ILL_DEFINED','SPICULATED'],
                             'pathology', topk).to(self.whole_trainer.device)
                self.whole_trainer.public_tensors['train_pred_ccce_gt'].append(pred_ccce_gt)
                # print("train_pred_ccce_gt:")
                # print(pred_ccce_gt)

    def _train_subepoch(self, epoch, subepoch):
        self.a2l_atter.train()
        self.train_metrics.reset()
        self.train_losses.reset()
        self.tmp_modified_att['train_modified_att'] = []
        for batch_idx, (_, _, ccce_gt, _, _) in enumerate(self.whole_trainer.train_data_loader):
            # train_ccce = ccce_gt.to(self.whole_trainer.device)
            train_ccce = self.whole_trainer.public_tensors['train_pred_ccce_gt'][batch_idx]#todo by ljw 20240702

            train_att = self.whole_trainer.public_tensors['train_i2a_att_feat'][batch_idx]
            train_att = train_att.reshape(train_att.size(0), -1).to(self.whole_trainer.device)  # by ljw 20240625 (B,6,C)->(B,6*C)

            self.optim_a2l_atter.zero_grad()
            self.a2l_imp.reset(train_att)
            inner_log = self.a2l_imp.solve()
            diff = self.a2l_imp.att - train_att  # diff between att* and  att_hat (B,C*6) C=16

            diff_reshape=diff.reshape(train_ccce.size(0),6,-1).to(self.whole_trainer.device)#by ljw 20240627 diff (B,6*C)->(B,6,C)
            train_ccce_expanded=train_ccce.unsqueeze(-1).repeat(1, 1,diff_reshape.size(2)).to(self.whole_trainer.device) #by ljw 20240627 (B,6)->(B,6,1)->(B,6,C)
            alignloss = self.a2l_atter_loss(diff_reshape, train_ccce_expanded)*1e3#todo by ljw add 1e3,align_loss is too small
            alignloss.backward()

            # ##todo update test,don't forget to delete it
            # print(f"A2L atter subepoch{subepoch} before")
            # self.a2l_atter.print_conv_weights()
            # self.optim_a2l_atter.step()
            # print(f"A2L atter subepoch{subepoch} after")
            # self.a2l_atter.print_conv_weights()
            # print("")

            self.optim_a2l_atter.step()

            # todo ask minghzou!!!! by ljw 20240627  get modified_att from features by i2a_clser
            att_pred = self.whole_trainer.i2a_clser.forward_feat2att(
                self.a2l_imp.att)  # self.a2l_imp.att(feature) match with att_pred (attributes classification),(B,2),0~1 float prob
            att0_pred = self.whole_trainer.i2a_clser.forward_feat2att(
                self.a2l_imp.att0)  # self.a2l_imp.att0(feature) match with att0_pred (attributes classification),(B,2),0~1 float prob
            modified_att = modify_att(att0_pred, att_pred)
            self.tmp_modified_att['train_modified_att'].append(modified_att)

            with torch.no_grad():
                if self.thresh_hold<1:
                    a2l_attention_mask = digitize(
                    self.a2l_atter(train_att.to(self.whole_trainer.device).float().detach().clone()),self.thresh_hold)#todo by ljw 20240702
                else:
                    a2l_attention_mask = get_topk(
                        self.a2l_atter(train_att.to(self.whole_trainer.device).float().detach().clone()),C=16,topk=self.thresh_hold)  # todo by ljw 20240702
                # print("train a2l ccce pred:")
                # print(a2l_attention_mask[:,::16])
                global_step = (epoch * self.a2l_atter_subepochs + subepoch) * len(
                    self.whole_trainer.train_data_loader) + batch_idx
                self.whole_trainer.writer.set_step(global_step)

                idx=0
                for met in self.a2l_atter_losses:
                    self.train_losses.update(self.train_losses.keys[idx], met(diff_reshape, train_ccce_expanded).item())
                    idx+=1
                idx=0
                for met in self.a2l_atter_metrics:
                    self.train_metrics.update(self.train_metrics.keys[idx], met(a2l_attention_mask, train_ccce_expanded.reshape(train_ccce_expanded.size(0),-1)))
                    idx+=1

        log_metrics = self.train_metrics.result()
        log_losses = self.train_losses.result()
        log = {**log_metrics, **log_losses}  # merge 2 dicts
        if self.do_valation:
            val_log = self._val_subepoch(epoch, subepoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
        return log

    def _val_subepoch(self, epoch, subepoch):
        self.a2l_atter.eval()
        self.val_metrics.reset()
        self.val_losses.reset()
        self.tmp_modified_att['val_modified_att'] = []

        for batch_idx, (_, _, ccce_gt, _, _) in enumerate(self.whole_trainer.val_data_loader):
            # val_ccce = ccce_gt.to(self.whole_trainer.device)
            val_ccce = self.whole_trainer.public_tensors['val_pred_ccce_gt'][batch_idx]  # todo by ljw 20240702

            val_att = self.whole_trainer.public_tensors['val_i2a_att_feat'][batch_idx].to(self.whole_trainer.device)
            val_att = val_att.reshape(val_att.size(0), -1)  # by ljw 20240625 (B,6,C)->(B,6*C)
            with torch.no_grad():
                self.a2l_imp.reset(val_att)
                inner_log = self.a2l_imp.solve()
                diff = self.a2l_imp.att - val_att  # diff between att* and  att_hat

                diff_reshape = diff.reshape(val_ccce.size(0), 6, -1).to(self.whole_trainer.device)  # by ljw 20240627 diff (B,6*C)->(B,6,C)
                val_ccce_expanded = val_ccce.unsqueeze(-1).repeat(1, 1, diff_reshape.size(
                    2)).to(self.whole_trainer.device)  # by ljw 20240627 (B,6)->(B,6,1)->(B,6,C)

                # todo ask minghzou!!!! by ljw 20240627  get modified_att from features by i2a_clser
                att_pred = self.whole_trainer.i2a_clser.forward_feat2att(self.a2l_imp.att)
                att0_pred = self.whole_trainer.i2a_clser.forward_feat2att(self.a2l_imp.att0)
                modified_att = modify_att(att0_pred, att_pred)

                self.tmp_modified_att['val_modified_att'].append(modified_att)
                if self.thresh_hold<1:
                    a2l_attention_mask = digitize(self.a2l_atter(val_att.to(self.whole_trainer.device)).float().detach().clone(),self.thresh_hold)#todo by ljw 20240704
                else:
                    a2l_attention_mask = get_topk(self.a2l_atter(val_att.to(self.whole_trainer.device)).float().detach().clone(),C=16,topk=self.thresh_hold)  # todo by ljw 20240704


                # print("val a2l ccce pred:")
                # print(a2l_attention_mask)
                # print(a2l_attention_mask[:, ::16])

                global_step = (epoch * self.a2l_atter_subepochs + subepoch) * len(
                    self.whole_trainer.val_data_loader) + batch_idx
                self.whole_trainer.writer.set_step(global_step, 'val')

                idx=0
                for met in self.a2l_atter_losses:
                    self.val_losses.update(self.val_losses.keys[idx], met(diff_reshape,val_ccce_expanded).item())
                    idx+=1
                idx=0
                for met in self.a2l_atter_metrics:
                    self.val_metrics.update(self.val_metrics.keys[idx], met(a2l_attention_mask,val_ccce_expanded.reshape(val_ccce_expanded.size(0),-1)))
                    idx+=1

        log_metrics = self.val_metrics.result()
        log_losses = self.val_losses.result()
        return {**log_metrics, **log_losses}  # merge 2 dicts

    def train(self, epoch):
        """
        Full training logic
        for subepoch in range(0, self.a2l_atter_subepochs):
            train subepoch,val_subepoch
            evaluate model performance according to configured metric,updpate best subepoch
            save every subepoch checkpoint and update the best subepoch checkpoint
            ues best subepoch atter in this epoch,record train_a2l_attention_mask and train_modified_att
        load best subepoch clser in this epoch, for i2a_atter imp_solver and i2a_att_pred record
        """
        if self.if_ccce_match_att ==1:#todo by ljw 20240702 ccce_match
            self.get_pred_match_ccce()

        self.att_tosave = dict()
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf  # get best subepoch chkeckppoint in this epoch
        for subepoch in tqdm(range(0, self.a2l_atter_subepochs),
                             desc='gpu:{} Epoch[a2l_atter]: {}'.format(self.whole_trainer.config['gpu'], epoch)):
            result = self._train_subepoch(epoch, subepoch)
            # save logged informations into log dict
            log = {'a2l_atter_subepochs': subepoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.whole_trainer.logger.info('    {:3s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.whole_trainer.logger.warning("Warning: Metric '{}' is not found. "
                                                      "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    self.best_log = log
                    best = True

            # save all subepochs checkpoint and the best subepoch checkpoint
            self.att_tosave['subepoch:{}'.format(subepoch)] = deepcopy(self.a2l_atter.state_dict())
            if subepoch % 20 == 0:
                torch.save(self.att_tosave, os.path.join(self.checkpoint_dir, 'a2l_att_epoch:{}.pt'.format(epoch)))
            if best:
                self.whole_trainer.public_tensors[
                    'a2l_atter'] = self.a2l_atter.state_dict()  # save best pt in this epoch
                torch.save(self.whole_trainer.public_tensors['a2l_atter'],
                           os.path.join(self.checkpoint_dir, 'a2l_att_epoch:{}_best.pt'.format(epoch)))

                with torch.no_grad():  # use this epoch best atter,update public_tensors['train_a2l_attention_mask'] public_tensors['train_modified_att']
                    self.whole_trainer.public_tensors['train_a2l_attention_mask'] = []
                    self.whole_trainer.public_tensors['train_modified_att'] = []
                    self.whole_trainer.public_tensors['val_a2l_attention_mask'] = []
                    self.whole_trainer.public_tensors['val_modified_att'] = []

                    for batch_idx, (_, _, _, _, _) in enumerate(self.whole_trainer.val_data_loader):
                        val_att = self.whole_trainer.public_tensors['val_i2a_att_feat'][batch_idx].to(
                            self.whole_trainer.device)
                        val_att = val_att.reshape(val_att.size(0), -1)  # by ljw 20240625 (B,6,C)->(B,6*C)

                        if self.thresh_hold < 1:
                            a2l_attention_mask = digitize(
                                self.a2l_atter(val_att.to(self.whole_trainer.device)).float().detach().clone(),
                                self.thresh_hold) # todo by ljw 20240704
                        else:
                            a2l_attention_mask = get_topk(
                                self.a2l_atter(val_att.to(self.whole_trainer.device)).float().detach().clone(), C=16,
                                topk=self.thresh_hold)  # todo by ljw 20240704

                        self.whole_trainer.public_tensors['val_a2l_attention_mask'].append(a2l_attention_mask)

                    self.whole_trainer.public_tensors['val_modified_att'] = deepcopy(
                        self.tmp_modified_att['val_modified_att'])  # use tmp_modified_att,avoid call i2a_imp here will affect atter parameters

                    for batch_idx, (_, _, _, _, _) in enumerate(self.whole_trainer.train_data_loader):
                        train_att = self.whole_trainer.public_tensors['train_i2a_att_feat'][batch_idx].to(
                            self.whole_trainer.device)
                        train_att = train_att.reshape(train_att.size(0), -1)  # by ljw 20240625 (B,6,C)->(B,6*C)

                        if self.thresh_hold < 1:
                            a2l_attention_mask = digitize(
                                self.a2l_atter(train_att.to(self.whole_trainer.device)).float().detach().clone(),
                                self.thresh_hold) # todo by ljw 20240704
                        else:
                            a2l_attention_mask = get_topk(
                                self.a2l_atter(train_att.to(self.whole_trainer.device)).float().detach().clone(), C=16,
                                topk=self.thresh_hold)  # todo by ljw 20240704


                        self.whole_trainer.public_tensors['train_a2l_attention_mask'].append(a2l_attention_mask)

                    self.whole_trainer.public_tensors['train_modified_att'] = deepcopy(
                        self.tmp_modified_att['train_modified_att'])

        self.a2l_atter.load_state_dict(
            self.whole_trainer.public_tensors['a2l_atter'])  # load best pt for a2l_atter imp_solver


class I2A_Atter_Trainer:
    def __init__(self, i2a_atter, i2a_imp, optim_i2a_atter, i2a_atter_metrics, i2a_atter_losses, i2a_atter_loss,
                 whole_trainer):
        self.i2a_atter = i2a_atter
        self.i2a_imp = i2a_imp
        self.i2a_atter_loss = i2a_atter_loss
        self.i2a_atter_metrics = i2a_atter_metrics
        self.i2a_atter_losses = i2a_atter_losses
        self.optim_i2a_atter = optim_i2a_atter
        self.i2a_att_mult = whole_trainer.config['i2a_atter']['mult'],  # by ljw 20240731
        self.i2a_atter_subepochs = whole_trainer.config["i2a_atter"]["subepochs"]
        self.monitor = whole_trainer.config["i2a_atter"]["monitor"]

        self.mnt_mode, self.mnt_metric = self.monitor.split()
        assert self.mnt_mode in ['min', 'max']
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf

        self.checkpoint_dir = os.path.join(whole_trainer.config.save_dir, 'i2a_atter')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.att_tosave = dict()

        self.whole_trainer = whole_trainer  # whole_trainer will bring public data_loader,logger,device,writer and publis_tensors
        self.do_valation = self.whole_trainer.val_data_loader is not None

        # metris writer should follow
        metric_names = ['i2a_prec', 'i2a_rec', 'i2a_dice']
        self.train_metrics = MetricTracker(*metric_names, writer=self.whole_trainer.writer)
        self.val_metrics = MetricTracker(*metric_names,  writer=self.whole_trainer.writer)
        # losses writer should follow
        loss_names = ['i2a_reward', 'i2a_punish', 'i2a_align_ratio']
        self.train_losses = MetricTracker(*loss_names, writer=self.whole_trainer.writer)
        self.val_losses = MetricTracker(*loss_names, writer=self.whole_trainer.writer)

    def _train_subepoch(self, epoch, subepoch):
        self.i2a_atter.train()
        self.train_metrics.reset()
        self.train_losses.reset()

        # self.mask_gt[ind],self.att_gt[ind],self.ccce_gt[ind],self.z[ind],self.y[ind]
        for batch_idx, (mask_gt, _, _, z, _) in enumerate(self.whole_trainer.train_data_loader):
            train_z = z.to(self.whole_trainer.device)
            train_mask = mask_gt.to(self.whole_trainer.device)
            modified_att = self.whole_trainer.public_tensors['train_modified_att'][batch_idx].to(
                self.whole_trainer.device)

            self.optim_i2a_atter.zero_grad()
            self.i2a_imp.reset(train_z, modified_att)
            inner_log = self.i2a_imp.solve()  # imp inner loop
            diff = self.i2a_imp.z - train_z  # difference between z* and z
            alignloss = self.i2a_atter_loss(diff, train_mask)*(torch.tensor(self.i2a_att_mult).to(self.whole_trainer.device))#todo by ljw 20240731
            alignloss.backward()

            # # #todo update test,don't forget delete it
            # print(f"I2A atter subepoch{subepoch} before")
            # self.i2a_atter.print_conv_weights()
            # self.optim_i2a_atter.step()
            # print(f"I2A atter subepoch{subepoch} after")
            # self.i2a_atter.print_conv_weights()
            # print()

            self.optim_i2a_atter.step()

            with torch.no_grad():
                i2a_attention_mask = digitize(
                    self.i2a_atter(train_z.to(self.whole_trainer.device)).float().detach().clone())

                global_step = (epoch * self.i2a_atter_subepochs + subepoch) * len(
                    self.whole_trainer.train_data_loader) + batch_idx
                self.whole_trainer.writer.set_step(global_step)

                idx=0
                for met in self.i2a_atter_losses:
                    self.train_losses.update(self.train_losses.keys[idx], met(diff, train_mask).item())
                    idx+=1
                idx=0
                for met in self.i2a_atter_metrics:
                    self.train_metrics.update(self.train_metrics.keys[idx], met(i2a_attention_mask, train_mask))
                    idx+=1


        log_metrics = self.train_metrics.result()
        log_losses = self.train_losses.result()
        log = {**log_metrics, **log_losses}  # merge 2 dicts
        if self.do_valation:
            val_log = self._val_subepoch(epoch, subepoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
        return log

    def _val_subepoch(self, epoch, subepoch):
        self.i2a_atter.eval()
        self.val_metrics.reset()
        self.val_losses.reset()

        for batch_idx, (mask_gt, _, _, z, _) in enumerate(self.whole_trainer.val_data_loader):
            val_z = z.to(self.whole_trainer.device)
            val_mask = mask_gt.to(self.whole_trainer.device)
            modified_att = self.whole_trainer.public_tensors['val_modified_att'][batch_idx].to(
                self.whole_trainer.device)

            with torch.no_grad():
                self.i2a_imp.reset(val_z, modified_att)
                inner_log = self.i2a_imp.solve()  # imp inner loop
                diff = self.i2a_imp.z - val_z  # difference between z* and z

                i2a_attention_mask = digitize(
                    self.i2a_atter(val_z.to(self.whole_trainer.device)).float().detach().clone())

                global_step = (epoch * self.i2a_atter_subepochs + subepoch) * len(
                    self.whole_trainer.val_data_loader) + batch_idx
                self.whole_trainer.writer.set_step(global_step, 'val')

                idx=0
                for met in self.i2a_atter_losses:
                    self.val_losses.update(self.val_losses.keys[idx], met(diff, val_mask).item())
                    idx+=1
                idx=0
                for met in self.i2a_atter_metrics:
                    self.val_metrics.update(self.val_metrics.keys[idx], met(i2a_attention_mask, val_mask))
                    idx+=1

        log_metrics = self.val_metrics.result()
        log_losses = self.val_losses.result()
        return {**log_metrics, **log_losses}  # merge 2 dicts

    def train(self, epoch):
        """
        Full training logic
        for subepoch in range(0, self.i2a_atter_subepochs):
            train subepoch,val_subepoch
            evaluate model performance according to configured metric,updpate best subepoch
            save every subepoch checkpoint and update the best subepoch checkpoint
            ues best subepoch atter in this epoch,record train_i2a_attention_mask and train_modified_att
        load best subepoch clser in this epoch, for i2a_atter imp_solver and i2a_att_pred record
        """
        self.att_tosave = dict()
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf#get best subepoch chkeckppoint in this epoch
        for subepoch in tqdm(range(0, self.i2a_atter_subepochs),
                             desc='gpu:{} Epoch[i2a_atter]: {}'.format(self.whole_trainer.config['gpu'], epoch)):
            result = self._train_subepoch(epoch, subepoch)
            # save logged informations into log dict
            log = {'i2a_atter_subepochs': subepoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.whole_trainer.logger.info('    {:3s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.whole_trainer.logger.warning("Warning: Metric '{}' is not found. "
                                                      "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    self.best_log = log
                    best = True

            # save all subepochs checkpoint and the best subepoch checkpoint
            self.att_tosave['subepoch:{}'.format(subepoch)] = deepcopy(self.i2a_atter.state_dict())
            if subepoch % 20 == 0:
                torch.save(self.att_tosave, os.path.join(self.checkpoint_dir, 'i2a_att_epoch:{}.pt'.format(epoch)))
            if best:
                self.whole_trainer.public_tensors[
                    'i2a_atter'] = self.i2a_atter.state_dict()  # save best pt in this epoch
                torch.save(self.whole_trainer.public_tensors['i2a_atter'],
                           os.path.join(self.checkpoint_dir, 'i2a_att_epoch:{}_best.pt'.format(epoch)))

                with torch.no_grad():  # use this epoch best atter,update public_tensors['train_i2a_attention_mask'] public_tensors['train_modified_att']
                    self.whole_trainer.public_tensors['train_i2a_attention_mask'] = []
                    self.whole_trainer.public_tensors['val_i2a_attention_mask'] = []

                    for batch_idx, (_, _, _, z, _) in enumerate(self.whole_trainer.val_data_loader):
                        val_z = z.to(self.whole_trainer.device)
                        i2a_attention_mask = digitize(
                            self.i2a_atter(val_z.to(self.whole_trainer.device)).float().detach().clone())
                        self.whole_trainer.public_tensors['val_i2a_attention_mask'].append(i2a_attention_mask)
                    for batch_idx, (_, _, _, z, _) in enumerate(self.whole_trainer.train_data_loader):
                        train_z = z.to(self.whole_trainer.device)
                        i2a_attention_mask = digitize(
                            self.i2a_atter(train_z.to(self.whole_trainer.device)).float().detach().clone())
                        self.whole_trainer.public_tensors['train_i2a_attention_mask'].append(i2a_attention_mask)

        self.i2a_atter.load_state_dict(
            self.whole_trainer.public_tensors['i2a_atter'])  # load best pt for i2a_atter imp_solver

















