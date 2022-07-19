import random
import torch
import torch.backends.cudnn as cudnn

import os
import numpy as np

from utils import AverageMeter, Accuracy, ProgressMeter, MeanTopKRecallMeter_domain, \
    Prototype, ISDALoss

from config import config_parser, cfg, update_config
from utils import get_data, get_model, get_loss

class Trainer:
    def __init__(self):
        args = config_parser.parse_args()
        update_config(cfg, args)
        self.cfg = cfg

        self.device = torch.device("cuda")
        self.dtype = torch.float32

        random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        cudnn.deterministic = True

        # Data
        self.train_data, self.val_data, self.test_data = get_data(cfg)

        # Model, Optimizer, LR Scheduler
        self.model, self.optimizer, self.scheduler = get_model(cfg)

        # Loss
        label_set = []
        for dataloader in self.train_data:
            train_label = dataloader.data_loader.dataset.label
            label_set.append(train_label)

        self.loss_f = get_loss(label_set, cfg.DATA.CLASS, cfg)
        self.prototype = Prototype(self.model.dim_f, cfg.DATA.CLASS, cfg.DATA.DOMAIN-1)
        self.loss_aug_f = ISDALoss(self.model.dim_f, cfg.DATA.CLASS, self.model.s.detach())

        # Save Configurations
        self.get_trial()

    def get_trial(self):
        self.trial_name = "{}" \
            .format(self.cfg.NAME)

        if os.path.isabs(self.cfg.OUTPUT_DIR):
            self.savedir = self.cfg.OUTPUT_DIR
        else:
            self.savedir = '../' + self.cfg.OUTPUT_DIR

        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)

        with open(self.savedir + self.trial_name + ".yaml", "w") as f:
            f.write(cfg.dump())  # save config to file

    def save_model(self, epoch, best_result):
        # save the model and optimizer in checkpoint
        torch.save({
            'state_dict': self.model.state_dict(),
            'epoch': epoch,
            'best_result': best_result,
            'scheduler': self.scheduler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, self.savedir + self.trial_name + ".pth")

    def resume_model(self):
        checkpoint = torch.load('../checkpoint/' + self.trial_name + ".pth")
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train(self):
        # Before training
        self.best_val_acc = 0. # used for selecting best model
        self.best_test_acc = 0. # just for monitoring, not for selecting best model
        self.best_test_unseen_acc = 0. # just for monitoring, not for selecting best models

        # Start Training
        for epoch in range(self.cfg.TRAIN.MAX_EPOCH):
            self.train_one_epoch(epoch)
            self.val_one_epoch(epoch)


    def train_one_epoch(self, epoch):
        # metric meter
        cls_meter = AverageMeter('C Loss', ':3.2f')
        s2v_meter = AverageMeter('S2V Loss', ':3.2f')
        v2s_meter = AverageMeter('V2S Loss', ':3.2f')
        s2s_meter = AverageMeter('S2S Loss', ':3.2f')
        aug_meter = AverageMeter('Aug Loss', ':3.2f')

        meta_cls_meter = AverageMeter('C Loss', ':3.2f')
        meta_v2s_meter = AverageMeter('Meta_V2S Loss', ':3.2f')
        meta_cross_meter = AverageMeter('Meta_Cross Loss', ':3.2f')
        meta_aug_meter = AverageMeter('Meta Aug Loss', ':3.2f')

        progress = ProgressMeter(
            self.cfg.TRAIN.ITER_PER_EPOCH,
            [cls_meter, s2v_meter, v2s_meter, s2s_meter, aug_meter,
             meta_cls_meter, meta_v2s_meter, meta_cross_meter, meta_aug_meter],
            prefix="Epoch: [{}]".format(epoch))

        # step
        self.model.train()

        tgt_all =[]
        d_all = []
        f_all = []

        for i in range(cfg.TRAIN.ITER_PER_EPOCH):
            x_s_all = []
            y_s_all = []
            tgt_s_all = []
            f_s_all = []

            self.prototype.freeze()

            # online generate domian split
            domain_split = torch.randperm(cfg.DATA.DOMAIN-1)

            self.model.zero_grad()

            # ------------------ Meta Train ----------------------- #
            fast_parameters = list(self.model.parameters())
            #print(fast_parameters)

            for weight in self.model.parameters():
                weight.fast = None
            self.model.zero_grad()

            cls_loss = []
            v2s_loss = []
            aug_loss = []

            for domain in domain_split[:-1]:
                x_s, tgt_s, _ = next(self.train_data[domain])
                x_s = x_s.to(self.device, self.dtype)
                tgt_s = tgt_s.to(self.device)

                y_s, f_s = self.model(x_s)
                cls_loss.append(self.loss_f[domain](y_s, tgt_s))
                v2s_loss.append(self.model.mapping_v2s(f_s, tgt_s, self.loss_f[domain]))

                if epoch >= self.cfg.AUG.EPOCH:
                    self.loss_aug_f.estimator.update_CV(f_s.detach(), tgt_s)
                    aug_loss.append(self.loss_aug_f(self.model, f_s, y_s, tgt_s,
                                                    self.cfg.AUG.LAM * (epoch - self.cfg.AUG.EPOCH) / (
                                                                self.cfg.TRAIN.MAX_EPOCH - self.cfg.AUG.EPOCH),
                                                    self.loss_f[domain]))
                else:
                    aug_loss.append(torch.FloatTensor([0.]).to(self.device))

                x_s_all.append(x_s)
                tgt_s_all.append(tgt_s)
                y_s_all.append(y_s)
                f_s_all.append(f_s)
                self.prototype.update_statistics(f_s, tgt_s, torch.ones_like(tgt_s) * domain)

                tgt_all.append(tgt_s.cpu().data.numpy())
                f_all.append(f_s.cpu().data.numpy())
                d_all.append(domain * torch.ones_like(tgt_s).cpu().data.numpy())

            x_s_all = torch.cat(x_s_all, dim=0)
            y_s_all = torch.cat(y_s_all, dim=0)
            f_s_all = torch.cat(f_s_all, dim=0)
            tgt_s_all = torch.cat(tgt_s_all, dim=0)

            cls_loss = torch.mean(torch.stack(cls_loss))
            v2s_loss = torch.mean(torch.stack(v2s_loss))
            aug_loss = torch.mean(torch.stack(aug_loss))

            s2s_loss = self.model.mapping_s2s(domain_split[:-1], self.prototype)
            s2v_loss = self.model.mapping_s2v(domain_split[:-1], self.prototype)

            meta_train_loss  = cls_loss + \
                               self.cfg.W.V2S * v2s_loss + self.cfg.W.S2S * s2s_loss + self.cfg.W.S2V * s2v_loss +\
                               self.cfg.W.AUG * aug_loss

            # ----------------------- Meta Test -----------------------------------
            grad = torch.autograd.grad(meta_train_loss, fast_parameters,
                                       create_graph=True, allow_unused=True)

            if self.cfg.META.STOP_GRADIENT:
                grad = [g.detach() for g in
                        grad]

            fast_parameters = []

            torch.nn.utils.clip_grad_norm_(grad, 2.)
            for k, weight in enumerate(self.model.parameters()):
                if weight.fast is None:  # this is to enable fast parameter
                    weight.fast = weight - cfg.META.META_STEP_SIZE * grad[k]
                else:
                    weight.fast = weight.fast - cfg.META.META_STEP_SIZE * grad[k]
                fast_parameters.append(
                    weight.fast)

            x_s, tgt_s, _ = next(self.train_data[domain_split[-1]])
            x_s = x_s.to(self.device, self.dtype)
            tgt_s = tgt_s.to(self.device)
            y_s, f_s = self.model(x_s)

            meta_cls_loss = self.loss_f[domain_split[-1]](y_s, tgt_s)
            meta_v2s_loss = self.model.mapping_v2s(f_s, tgt_s, self.loss_f[domain_split[-1]])
            meta_cross_loss = self.model.cross_domain_prototype(f_s, domain_split[-1], tgt_s, self.prototype,
                                                           self.loss_f[domain_split[-1]])
            if epoch >= self.cfg.AUG.EPOCH:
                meta_aug_loss = self.loss_aug_f(self.model, f_s, y_s, tgt_s,
                                                self.cfg.AUG.LAM * (epoch - self.cfg.EPOCH_Aug) / (
                                                                self.cfg.MAX_EPOCH - self.cfg.EPOCH_Aug),
                                                self.loss_f[domain_split[-1]])
            else:
                meta_aug_loss = torch.FloatTensor([0.]).cuda()

            meta_val_loss = meta_cls_loss + + self.cfg.W.V2S_M * (meta_v2s_loss+meta_cross_loss) + self.cfg.W.AUG_M * meta_aug_loss

            ## update metric
            cls_acc = Accuracy(y_s_all, tgt_s_all)[0]

            cls_meter.update(cls_loss.item(), x_s_all.size(0))
            v2s_meter.update(v2s_loss.item(), x_s_all.size(0))
            s2s_meter.update(s2s_loss.item(), x_s_all.size(0))
            s2v_meter.update(s2v_loss.item(), x_s_all.size(0))
            aug_meter.update(aug_loss.item(), x_s_all.size(0))

            meta_cls_meter.update(meta_cls_loss.item(), x_s_all.size(0))
            meta_v2s_meter.update(meta_v2s_loss.item(), x_s_all.size(0))
            meta_cross_meter.update(meta_cross_loss.item(), x_s_all.size(0))
            meta_aug_meter.update(meta_aug_loss.item(), x_s_all.size(0))

            # Update
            self.optimizer.zero_grad()
            total_loss = meta_train_loss + self.cfg.W.VAL * meta_val_loss
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if i % self.cfg.TRAIN.PRINT_FREQ == 0:
                progress.display(i)

    def val_one_epoch(self , epoch):
        # metric meter
        metric_meter = MeanTopKRecallMeter_domain(num_domain=self.cfg.DATA.DOMAIN-1, num_classes=self.cfg.DATA.CLASS)

        f_all = []
        d_all = []
        x_all = []
        y_all = []
        tgt_all = []

        for weight in self.model.parameters():
            weight.fast = None

        self.model.eval()

        for domain, val_loader in enumerate(self.val_data):
            for i, (x_s, tgt_s, _) in enumerate(val_loader):
                x_s = x_s.cuda()
                tgt_s = tgt_s.cuda()

                ## compute output
                with torch.no_grad():
                    y_s, f_s = self.model(x_s)

                metric_meter.add(y_s.cpu().data.numpy(),
                                 tgt_s.cpu().data.numpy(),
                                 domain * torch.ones_like(tgt_s).cpu().data.numpy())

            # cls_acc = accuracy(y_s_all, tgt_s_all)[0]
            print('domain %d acc %.3f.' % (domain, metric_meter.domain_aware_acc(domain)))

        print('Mean Val acc %.3f. [%.3f]' % (metric_meter.domain_mean_acc(), max(metric_meter.domain_mean_acc(), self.best_val_acc)))

        if self.cfg.TRAIN.PRINT_VAL:
            metric_meter.domain_class_aware_recall()

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
