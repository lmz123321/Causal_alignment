import os
from copy import deepcopy

import torch
from abc import abstractmethod

import torchopt
from numpy import inf
import sys
sys.path.append("/home/lijingwen/Projects")
import Counter_align.LIDC.loss.loss as module_losses
import Counter_align.LIDC.utils.metrics as module_metrics
import Counter_align.LIDC.data_loader.data_loaders as module_data
from Counter_align.LIDC.implicit_gradient.image2label import get_implicit_i2lmodel
import Counter_align.LIDC.model.attention as module_atter
import Counter_align.LIDC.model.classifier as module_clser
from Counter_align.LIDC.utils.util import att_preds_totensor, digitize, MetricTracker, modify_att, get_topk
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
            training=False, valid=True, shuffle=False, seed=config['seed'],
            batch_size=config['data_loader']['args']['batch_size'],
        )
        self.test_data_loader = getattr(module_data, config['data_loader']['type'])(
            cache_path=config['data_loader']['args']['cache_path'],
            ccce_file=config['data_loader']['args']['ccce_file'],
            training=False, valid=False, shuffle=False, seed=config['seed'],
            batch_size=config['data_loader']['args']['batch_size'],
        )

        self.logger.info('Each train/val epoch contains {}/{} steps in tensorboard, respectively'.format(
            len(self.train_data_loader), len(self.val_data_loader)))


        # i2l clser,atter,optimizer,solver
        self.i2l_clser = getattr(module_clser, config["i2l_clser"]["arch"])(in_dim=config["i2l_clser"]["in_dim"]).to(device)
        self.i2l_atter = getattr(module_atter, config["i2l_atter"]["arch"])(in_dim=config["i2l_atter"]["in_dim"]).to(device)

        self.optim_i2l_clser = torch.optim.Adam(self.i2l_clser.parameters(), lr=config["i2l_clser"]["lr"])
        self.optim_i2l_atter = torch.optim.Adam(self.i2l_atter.parameters(), lr=config["i2l_atter"]["lr"])
        self.i2l_solver = torchopt.linear_solve.solve_normal_cg(rtol=config["i2l_atter"]["rtol"],
                                                                maxiter=config["i2l_atter"]["maxiter"])
        self.i2l_imp = get_implicit_i2lmodel(self.i2l_solver)(self.i2l_clser, self.i2l_atter, 0.1)

        # metrics writer should follow
        self.i2l_clser_metrics = [getattr(module_metrics, met) for met in config["i2l_clser"]["metrics"]]
        self.i2l_atter_metrics = [getattr(module_metrics, met) for met in config["i2l_atter"]["metrics"]]

        # losses writer should follow
        self.i2l_clser_losses = [getattr(module_losses,loss) for loss in config["i2l_clser"]["losses"]]
        self.i2l_atter_losses = [getattr(module_losses,loss) for loss in config["i2l_atter"]["losses"]]

        # loss should monitor,save best chekpoint in this epoch based on these loss performance
        self.i2l_clser_loss = getattr(module_losses, config["i2l_clser"]["loss"])
        self.i2l_atter_loss = getattr(module_losses, config["i2l_atter"]["loss"])

        # public_tensors in 4 subepochs
        self.public_tensors = {'train_i2l_attention_mask': [mask_gt for mask_gt, _, _, _, _ in self.train_data_loader],
                               'val_i2l_attention_mask': [mask_gt for mask_gt, _, _, _, _ in self.val_data_loader],
                               'test_i2l_attention_mask': [mask_gt for mask_gt, _, _, _, _ in self.test_data_loader],
                               'train_i2l_att_feat': [], 'val_i2l_att_feat': [],'test_i2l_att_feat': [],
                               'i2l_clser': [], 'i2l_atter': [],
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
            ['i2l_cls_lr', self.config['i2l_clser']['lr']],
            ['i2l_cls_sub', self.config['i2l_clser']['subepochs']],
            ['i2l_att_lr', self.config['i2l_atter']['lr']],
            ['i2l_att_sub', self.config['i2l_atter']['subepochs']],
            ['i2l_att_lamb', self.config['i2l_atter']['lamb']],

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
        self.i2l_clser_trainer = i2l_Clser_Trainer(self.i2l_clser, self.optim_i2l_clser, self.i2l_clser_metrics,
                                                   self.i2l_clser_losses, self.i2l_clser_loss, self)

        self.i2l_atter_trainer = i2l_Atter_Trainer(self.i2l_atter, self.i2l_imp, self.optim_i2l_atter,
                                                   self.i2l_atter_metrics,self.i2l_atter_losses, self.i2l_atter_loss, self)

        self.save_code()
        for epoch in range(0, self.epochs):
            # print epoch information to the screen
            self.logger.info('    {:3s}: {}'.format('epoch', epoch))
            self.i2l_clser_trainer.train(epoch)
            self.i2l_atter_trainer.train(epoch)


    def save_code(self):
        shutil.copytree(str(self.codepath), str(self.code_dir), dirs_exist_ok=True)


class i2l_Clser_Trainer:
    '''
    Base class for trainers with alignment loss
    '''

    def __init__(self, i2l_clser, optim_i2l_clser, i2l_clser_metrics, i2l_clser_losses, i2l_clser_loss, whole_trainer):
        self.i2l_clser = i2l_clser
        self.i2l_clser_loss = i2l_clser_loss
        self.i2l_clser_metrics = i2l_clser_metrics
        self.i2l_clser_losses = i2l_clser_losses
        self.optim_i2l_clser = optim_i2l_clser
        self.i2l_clser_subepochs = whole_trainer.config["i2l_clser"]["subepochs"]
        self.monitor = whole_trainer.config["i2l_clser"]["monitor"]


        self.mnt_mode, self.mnt_metric = self.monitor.split()
        assert self.mnt_mode in ['min', 'max']
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf

        self.checkpoint_dir = os.path.join(whole_trainer.config.save_dir, 'i2l_clser')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.cls_tosave = dict()

        self.whole_trainer = whole_trainer  # whole_trainer will bring public data_loader,logger,device,writer and publis_tensors
        self.do_valation = self.whole_trainer.val_data_loader is not None

        # metris writer should follow
        metric_names = ['i2l_avg_acc']
        self.train_metrics = MetricTracker(*metric_names, writer=self.whole_trainer.writer)
        self.val_metrics = MetricTracker(*metric_names,writer=self.whole_trainer.writer)
        self.test_metrics = MetricTracker(*metric_names, writer=self.whole_trainer.writer)
        # losses writer should follow
        loss_names = ['i2l_celoss']
        self.train_losses = MetricTracker(*loss_names, writer=self.whole_trainer.writer)
        self.val_losses = MetricTracker(*loss_names, writer=self.whole_trainer.writer)
        self.test_losses = MetricTracker(*loss_names, writer=self.whole_trainer.writer)

    def _train_subepoch(self, epoch, subepoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        self.i2l_clser.train()
        self.train_metrics.reset()
        self.train_losses.reset()
        #self.mask_gt[ind],self.att_gt[ind],self.ccce_gt[ind],self.z[ind],self.y[ind]
        for batch_idx, (_, _, _, z, y) in enumerate(self.whole_trainer.train_data_loader):
            train_mask = self.whole_trainer.public_tensors['train_i2l_attention_mask'][batch_idx].to(
                self.whole_trainer.device)
            train_z = z.to(self.whole_trainer.device)
            train_y = y.long().unsqueeze(1).to(self.whole_trainer.device)
            self.optim_i2l_clser.zero_grad()
            preds = self.i2l_clser(train_mask * train_z)
            celoss = self.i2l_clser_loss(preds, train_y)
            celoss.backward()
            self.optim_i2l_clser.step()

            global_step=(epoch * self.i2l_clser_subepochs + subepoch)*len(self.whole_trainer.train_data_loader) + batch_idx
            self.whole_trainer.writer.set_step(global_step)

            self.train_losses.update('i2l_celoss', celoss.item())
            idx=0
            for met in self.i2l_clser_metrics:
                self.train_metrics.update(self.train_metrics.keys[idx], met(preds, train_y))
                idx+=1



        log_metrics = self.train_metrics.result()
        log_losses = self.train_losses.result()
        log = {**log_metrics, **log_losses}  # merge 2 dicts
        if self.do_valation:
            val_log = self._val_subepoch(epoch, subepoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

            test_log = self._test_subepoch(epoch, subepoch)
            log.update(**{'test_' + k: v for k, v in test_log.items()})

        return log
    def _test_subepoch(self, epoch, subepoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        self.i2l_clser.eval()
        self.test_metrics.reset()
        self.test_losses.reset()
        # self.mask_gt[ind],self.att_gt[ind],self.ccce_gt[ind],self.z[ind],self.y[ind]
        for batch_idx, (_, _, _, z, y) in enumerate(self.whole_trainer.test_data_loader):
            test_mask = self.whole_trainer.public_tensors['test_i2l_attention_mask'][batch_idx].to(
                self.whole_trainer.device)
            test_z = z.to(self.whole_trainer.device)
            test_y = y.unsqueeze(1).long().to(self.whole_trainer.device)
            with torch.no_grad():
                preds = self.i2l_clser(test_mask * test_z)
                celoss = self.i2l_clser_loss(preds, test_y)
            global_step = (epoch * self.i2l_clser_subepochs + subepoch) * len(self.whole_trainer.test_data_loader) + batch_idx
            self.whole_trainer.writer.set_step(global_step,'test')
            self.test_losses.update('i2l_celoss', celoss.item())
            idx = 0
            for met in self.i2l_clser_metrics:
                self.test_metrics.update(self.test_metrics.keys[idx], met(preds, test_y))
                idx += 1
        log_metrics = self.test_metrics.result()
        log_losses = self.test_losses.result()
        return {**log_metrics, **log_losses}  # merge 2 dicts

    def _val_subepoch(self, epoch, subepoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        self.i2l_clser.eval()
        self.val_metrics.reset()
        self.val_losses.reset()
        # self.mask_gt[ind],self.att_gt[ind],self.ccce_gt[ind],self.z[ind],self.y[ind]
        for batch_idx, (_, _, _, z, y) in enumerate(self.whole_trainer.val_data_loader):
            val_mask = self.whole_trainer.public_tensors['val_i2l_attention_mask'][batch_idx].to(
                self.whole_trainer.device)
            val_z = z.to(self.whole_trainer.device)
            val_y = y.unsqueeze(1).long().to(self.whole_trainer.device)
            with torch.no_grad():
                preds = self.i2l_clser(val_mask * val_z)
                celoss = self.i2l_clser_loss(preds, val_y)
            global_step = (epoch * self.i2l_clser_subepochs + subepoch) * len(self.whole_trainer.val_data_loader) + batch_idx
            self.whole_trainer.writer.set_step(global_step,'val')
            self.val_losses.update('i2l_celoss', celoss.item())
            idx = 0
            for met in self.i2l_clser_metrics:
                self.val_metrics.update(self.val_metrics.keys[idx], met(preds, val_y))
                idx += 1
        log_metrics = self.val_metrics.result()
        log_losses = self.val_losses.result()
        return {**log_metrics, **log_losses}  # merge 2 dicts

    def train(self, epoch):
        """
        Full training logic
        for subepoch in range(0, self.i2l_clser_subepochs):
            train subepoch,val_subepoch
            evaluate model performance according to configured metric,updpate best subepoch
            save every subepoch checkpoint and update the best subepoch checkpoint
            ues best subepoch clser in this epoch,record i2l_att_pred for a2l_clser
        load best subepoch clser in this epoch, for i2l_atter imp_solver and i2l_att_pred record

        """
        self.cls_tosave = dict()
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf  # get best subepoch chkeckppoint in this epoch
        for subepoch in tqdm(range(0, self.i2l_clser_subepochs),
                             desc='gpu:{} Epoch[i2l_cls]: {}'.format(self.whole_trainer.config['gpu'], epoch)):
            result = self._train_subepoch(epoch, subepoch)
            # save logged informations into log dict
            log = {'i2l_clser_subepochs': subepoch}
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
            self.cls_tosave['subepoch:{}'.format(subepoch)] = deepcopy(self.i2l_clser.state_dict())
            if subepoch % 20 == 0:
                torch.save(self.cls_tosave, os.path.join(self.checkpoint_dir, 'i2l_cls_epoch:{}.pt'.format(epoch)))
            if best:
                self.whole_trainer.public_tensors[
                    'i2l_clser'] = self.i2l_clser.state_dict()  # save best pt in this epoch
                torch.save(self.whole_trainer.public_tensors['i2l_clser'],
                           os.path.join(self.checkpoint_dir, 'i2l_cls_epoch:{}_best.pt'.format(epoch)))

                with torch.no_grad():  # use best clser for now,update public_tensors['train_i2l_att_feat'] public_tensors['val_i2l_att_feat']
                    self.whole_trainer.public_tensors['test_i2l_att_feat'] = []
                    self.whole_trainer.public_tensors['val_i2l_att_feat'] = []
                    self.whole_trainer.public_tensors['train_i2l_att_feat'] = []
                    self.whole_trainer.public_tensors['test_i2l_att_pred'] = []
                    self.whole_trainer.public_tensors['val_i2l_att_pred'] = []#todo by ljw 20240702
                    self.whole_trainer.public_tensors['train_i2l_att_pred'] = []#todo by ljw 20240702

                    for batch_idx, (_, _, _, z, _) in enumerate(self.whole_trainer.test_data_loader):
                        test_mask = self.whole_trainer.public_tensors['test_i2l_attention_mask'][batch_idx].to(
                            self.whole_trainer.device)
                        test_z = z.to(self.whole_trainer.device)
                        with torch.no_grad():
                            pred_feat = self.i2l_clser.forward_feature(test_mask * test_z)  # by ljw 20240625,pred_feat (B,6,C)
                            self.whole_trainer.public_tensors['test_i2l_att_feat'].append(
                                pred_feat)  #by ljw 20240625,pred change to feature format

                            preds = self.i2l_clser(test_mask * test_z)#todo by ljw 20240702
                            i2l_att_pred = att_preds_totensor(preds).detach().clone()#todo by ljw 20240702
                            self.whole_trainer.public_tensors['test_i2l_att_pred'].append(i2l_att_pred)#todo by ljw 20240702

                    for batch_idx, (_, _, _, z, _) in enumerate(self.whole_trainer.val_data_loader):
                        val_mask = self.whole_trainer.public_tensors['val_i2l_attention_mask'][batch_idx].to(
                            self.whole_trainer.device)
                        val_z = z.to(self.whole_trainer.device)
                        with torch.no_grad():
                            pred_feat = self.i2l_clser.forward_feature(val_mask * val_z)  # by ljw 20240625,pred_feat (B,6,C)
                            self.whole_trainer.public_tensors['val_i2l_att_feat'].append(
                                pred_feat)  #by ljw 20240625,pred change to feature format

                            preds = self.i2l_clser(val_mask * val_z)#todo by ljw 20240702
                            i2l_att_pred = att_preds_totensor(preds).detach().clone()#todo by ljw 20240702
                            self.whole_trainer.public_tensors['val_i2l_att_pred'].append(i2l_att_pred)#todo by ljw 20240702



                    for batch_idx, (_, _, _, z, _) in enumerate(self.whole_trainer.train_data_loader):
                        train_mask = self.whole_trainer.public_tensors['train_i2l_attention_mask'][batch_idx].to(
                            self.whole_trainer.device)
                        train_z = z.to(self.whole_trainer.device)
                        with torch.no_grad():
                            pred_feat=self.i2l_clser.forward_feature(train_mask * train_z)#by ljw 20240625,pred_feat (B,6,C)
                            self.whole_trainer.public_tensors['train_i2l_att_feat'].append(pred_feat)#by ljw 20240625,pred change to feature format

                            preds = self.i2l_clser(train_mask * train_z)#todo by ljw 20240702
                            i2l_att_pred = att_preds_totensor(preds).detach().clone()#todo by ljw 20240702
                            self.whole_trainer.public_tensors['train_i2l_att_pred'].append(i2l_att_pred)#todo by ljw 20240702


        self.i2l_clser.load_state_dict(self.whole_trainer.public_tensors[
                                           'i2l_clser'])  # load best pt for i2l_atter imp_solver and i2l_att_pred record




class i2l_Atter_Trainer:
    def __init__(self, i2l_atter, i2l_imp, optim_i2l_atter, i2l_atter_metrics, i2l_atter_losses, i2l_atter_loss,
                 whole_trainer):
        self.i2l_atter = i2l_atter
        self.i2l_imp = i2l_imp
        self.i2l_atter_loss = i2l_atter_loss
        self.i2l_atter_metrics = i2l_atter_metrics
        self.i2l_atter_losses = i2l_atter_losses
        self.optim_i2l_atter = optim_i2l_atter
        self.i2l_atter_subepochs = whole_trainer.config["i2l_atter"]["subepochs"]
        self.monitor = whole_trainer.config["i2l_atter"]["monitor"]

        self.mnt_mode, self.mnt_metric = self.monitor.split()
        assert self.mnt_mode in ['min', 'max']
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf

        self.checkpoint_dir = os.path.join(whole_trainer.config.save_dir, 'i2l_atter')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.att_tosave = dict()

        self.whole_trainer = whole_trainer  # whole_trainer will bring public data_loader,logger,device,writer and publis_tensors
        self.do_valation = self.whole_trainer.val_data_loader is not None

        # metris writer should follow
        metric_names = ['i2l_prec', 'i2l_rec', 'i2l_dice']
        self.train_metrics = MetricTracker(*metric_names, writer=self.whole_trainer.writer)
        self.val_metrics = MetricTracker(*metric_names,  writer=self.whole_trainer.writer)
        self.test_metrics = MetricTracker(*metric_names, writer=self.whole_trainer.writer)
        # losses writer should follow
        loss_names = ['i2l_reward', 'i2l_punish', 'i2l_align_ratio']
        self.train_losses = MetricTracker(*loss_names, writer=self.whole_trainer.writer)
        self.val_losses = MetricTracker(*loss_names, writer=self.whole_trainer.writer)
        self.test_losses = MetricTracker(*loss_names, writer=self.whole_trainer.writer)

    def _train_subepoch(self, epoch, subepoch):
        self.i2l_atter.train()
        self.train_metrics.reset()
        self.train_losses.reset()

        # self.mask_gt[ind],self.att_gt[ind],self.ccce_gt[ind],self.z[ind],self.y[ind]
        for batch_idx, (mask_gt, _, _, z, _) in enumerate(self.whole_trainer.train_data_loader):
            train_z = z.to(self.whole_trainer.device)
            train_mask = mask_gt.to(self.whole_trainer.device)

            self.optim_i2l_atter.zero_grad()
            self.i2l_imp.reset(train_z)
            inner_log = self.i2l_imp.solve()  # imp inner loop
            diff = self.i2l_imp.z - train_z  # difference between z* and z
            alignloss = self.i2l_atter_loss(diff, train_mask)
            alignloss.backward()

            # # #todo update test,don't forget delete it
            # print(f"i2l atter subepoch{subepoch} before")
            # self.i2l_atter.print_conv_weights()
            # self.optim_i2l_atter.step()
            # print(f"i2l atter subepoch{subepoch} after")
            # self.i2l_atter.print_conv_weights()
            # print()

            self.optim_i2l_atter.step()

            with torch.no_grad():
                i2l_attention_mask = digitize(
                    self.i2l_atter(train_z.to(self.whole_trainer.device)).float().detach().clone())

                global_step = (epoch * self.i2l_atter_subepochs + subepoch) * len(
                    self.whole_trainer.train_data_loader) + batch_idx
                self.whole_trainer.writer.set_step(global_step)

                idx=0
                for met in self.i2l_atter_losses:
                    self.train_losses.update(self.train_losses.keys[idx], met(diff, train_mask).item())
                    idx+=1
                idx=0
                for met in self.i2l_atter_metrics:
                    self.train_metrics.update(self.train_metrics.keys[idx], met(i2l_attention_mask, train_mask))
                    idx+=1


        log_metrics = self.train_metrics.result()
        log_losses = self.train_losses.result()
        log = {**log_metrics, **log_losses}  # merge 2 dicts
        if self.do_valation:
            val_log = self._val_subepoch(epoch, subepoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

            test_log = self._test_subepoch(epoch, subepoch)
            log.update(**{'test_' + k: v for k, v in test_log.items()})
        return log

    def _test_subepoch(self, epoch, subepoch):
        self.i2l_atter.eval()
        self.test_metrics.reset()
        self.test_losses.reset()

        for batch_idx, (mask_gt, _, _, z, _) in enumerate(self.whole_trainer.test_data_loader):
            test_z = z.to(self.whole_trainer.device)
            test_mask = mask_gt.to(self.whole_trainer.device)

            with torch.no_grad():
                self.i2l_imp.reset(test_z)
                inner_log = self.i2l_imp.solve()  # imp inner loop
                diff = self.i2l_imp.z - test_z  # difference between z* and z

                i2l_attention_mask = digitize(
                    self.i2l_atter(test_z.to(self.whole_trainer.device)).float().detach().clone())

                global_step = (epoch * self.i2l_atter_subepochs + subepoch) * len(
                    self.whole_trainer.test_data_loader) + batch_idx
                self.whole_trainer.writer.set_step(global_step, 'test')

                idx=0
                for met in self.i2l_atter_losses:
                    self.test_losses.update(self.test_losses.keys[idx], met(diff, test_mask).item())
                    idx+=1
                idx=0
                for met in self.i2l_atter_metrics:
                    self.test_metrics.update(self.test_metrics.keys[idx], met(i2l_attention_mask, test_mask))
                    idx+=1

        log_metrics = self.test_metrics.result()
        log_losses = self.test_losses.result()
        return {**log_metrics, **log_losses}  # merge 2 dicts

    def _val_subepoch(self, epoch, subepoch):
        self.i2l_atter.eval()
        self.val_metrics.reset()
        self.val_losses.reset()

        for batch_idx, (mask_gt, _, _, z, _) in enumerate(self.whole_trainer.val_data_loader):
            val_z = z.to(self.whole_trainer.device)
            val_mask = mask_gt.to(self.whole_trainer.device)

            with torch.no_grad():
                self.i2l_imp.reset(val_z)
                inner_log = self.i2l_imp.solve()  # imp inner loop
                diff = self.i2l_imp.z - val_z  # difference between z* and z

                i2l_attention_mask = digitize(
                    self.i2l_atter(val_z.to(self.whole_trainer.device)).float().detach().clone())

                global_step = (epoch * self.i2l_atter_subepochs + subepoch) * len(
                    self.whole_trainer.val_data_loader) + batch_idx
                self.whole_trainer.writer.set_step(global_step, 'val')

                idx=0
                for met in self.i2l_atter_losses:
                    self.val_losses.update(self.val_losses.keys[idx], met(diff, val_mask).item())
                    idx+=1
                idx=0
                for met in self.i2l_atter_metrics:
                    self.val_metrics.update(self.val_metrics.keys[idx], met(i2l_attention_mask, val_mask))
                    idx+=1

        log_metrics = self.val_metrics.result()
        log_losses = self.val_losses.result()
        return {**log_metrics, **log_losses}  # merge 2 dicts

    def train(self, epoch):
        """
        Full training logic
        for subepoch in range(0, self.i2l_atter_subepochs):
            train subepoch,val_subepoch
            evaluate model performance according to configured metric,updpate best subepoch
            save every subepoch checkpoint and update the best subepoch checkpoint
            ues best subepoch atter in this epoch,record train_i2l_attention_mask and train_modified_att
        load best subepoch clser in this epoch, for i2l_atter imp_solver and i2l_att_pred record
        """
        self.att_tosave = dict()
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf#get best subepoch chkeckppoint in this epoch
        for subepoch in tqdm(range(0, self.i2l_atter_subepochs),
                             desc='gpu:{} Epoch[i2l_atter]: {}'.format(self.whole_trainer.config['gpu'], epoch)):
            result = self._train_subepoch(epoch, subepoch)
            # save logged informations into log dict
            log = {'i2l_atter_subepochs': subepoch}
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
            self.att_tosave['subepoch:{}'.format(subepoch)] = deepcopy(self.i2l_atter.state_dict())
            if subepoch % 20 == 0:
                torch.save(self.att_tosave, os.path.join(self.checkpoint_dir, 'i2l_att_epoch:{}.pt'.format(epoch)))
            if best:
                self.whole_trainer.public_tensors[
                    'i2l_atter'] = self.i2l_atter.state_dict()  # save best pt in this epoch
                torch.save(self.whole_trainer.public_tensors['i2l_atter'],
                           os.path.join(self.checkpoint_dir, 'i2l_att_epoch:{}_best.pt'.format(epoch)))

                with torch.no_grad():  # use this epoch best atter,update public_tensors['train_i2l_attention_mask'] public_tensors['train_modified_att']
                    self.whole_trainer.public_tensors['train_i2l_attention_mask'] = []
                    self.whole_trainer.public_tensors['val_i2l_attention_mask'] = []
                    self.whole_trainer.public_tensors['test_i2l_attention_mask'] = []

                    for batch_idx, (_, _, _, z, _) in enumerate(self.whole_trainer.test_data_loader):
                        test_z = z.to(self.whole_trainer.device)
                        i2l_attention_mask = digitize(
                            self.i2l_atter(test_z.to(self.whole_trainer.device)).float().detach().clone())
                        self.whole_trainer.public_tensors['test_i2l_attention_mask'].append(i2l_attention_mask)

                    for batch_idx, (_, _, _, z, _) in enumerate(self.whole_trainer.val_data_loader):
                        val_z = z.to(self.whole_trainer.device)
                        i2l_attention_mask = digitize(
                            self.i2l_atter(val_z.to(self.whole_trainer.device)).float().detach().clone())
                        self.whole_trainer.public_tensors['val_i2l_attention_mask'].append(i2l_attention_mask)
                    for batch_idx, (_, _, _, z, _) in enumerate(self.whole_trainer.train_data_loader):
                        train_z = z.to(self.whole_trainer.device)
                        i2l_attention_mask = digitize(
                            self.i2l_atter(train_z.to(self.whole_trainer.device)).float().detach().clone())
                        self.whole_trainer.public_tensors['train_i2l_attention_mask'].append(i2l_attention_mask)

        self.i2l_atter.load_state_dict(
            self.whole_trainer.public_tensors['i2l_atter'])  # load best pt for i2l_atter imp_solver

















