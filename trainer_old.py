import random

import torch
import torch.backends.cudnn as cudnn

from torchvision import models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

import argparse
import os
import numpy as np

import data as datasets
from utils import ForeverDataIterator, AverageMeter, accuracy, ProgressMeter, MeanTopKRecallMeter_domain, \
    save_model, Domain_Alignment, class_counter, ISDALoss
import model as models
from torch.utils.data import WeightedRandomSampler
import utils
# from train_baseline import data_loading


def main(args):
    # set random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    # data preparation
    train_source_iter_list, train_source_balanced_iter_list, val_loader, test_loader = data_loading(args)

    # model prepration
    model, optimizer, lr_scheduler = model_loading(args)

    # prepare loss function
    loss_f = []
    for dataloader in train_source_iter_list:
        train_label = dataloader.data_loader.dataset.label
        # train_label = np.concatenate(train_label)
        args.num_class_list = class_counter(train_label, args.n_class)
        loss_f.append(utils.__dict__[args.loss](args).cuda())

    loss_aug_f = ISDALoss(512, args.n_class, embed=model.embed.detach())

    # train

    # start training
    best_val_acc = 0.
    best_test_acc = 0.
    best_test_unseen_acc = 0.

    '''
    print('Warm Up')
    for epoch in range(40):
        train_warm(train_source_iter_list, train_source_balanced_iter_list, model, loss_f, optimizer,
              lr_scheduler, epoch, args)
        print('Validation Set')
        val_acc, _ = val(val_loader, model, epoch, 'val', best_val_acc, args)
        print('Test Set')
        test_acc, test_unseen_acc = val(test_loader, model, epoch, 'test', best_test_acc, args)

        if val_acc > best_val_acc:
            print('\033[92m' + 'Val Current best acc' + '\033[0m')

            best_val_acc = max(val_acc, best_val_acc)
            save_model(model, args, 'val', args.trial)

        # remember best acc@1 and save checkpoint
        if test_acc > best_test_acc:
            print('\033[92m' + 'Test Current best acc' + '\033[0m')

            best_test_acc = max(test_acc, best_test_acc)
            save_model(model, args, 'test', args.trial)

        # remember best acc@1 and save checkpoint
        if test_unseen_acc > best_test_unseen_acc:
            print('\033[92m' + 'Test Unseen Current best acc' + '\033[0m')

            best_test_unseen_acc = max(test_unseen_acc, best_test_unseen_acc)
            save_model(model, args, 'test-unseen', args.trial)
    '''

    print('Start Training')
    for epoch in range(args.epochs):
        train(train_source_iter_list, train_source_balanced_iter_list, model, loss_f, loss_aug_f, optimizer,
              lr_scheduler, epoch, args)
        print('Validation Set')
        val_acc, _ = val(val_loader, model, epoch, 'val', best_val_acc, args)
        print('Test Set')
        test_acc, test_unseen_acc = val(test_loader, model, epoch, 'test', best_test_acc, args)

        if val_acc > best_val_acc:
            print('\033[92m' + 'Val Current best acc' + '\033[0m')

            best_val_acc = max(val_acc, best_val_acc)
            save_model(model, args, 'val', args.trial)

        # remember best acc@1 and save checkpoint
        if test_acc > best_test_acc:
            print('\033[92m' + 'Test Current best acc' + '\033[0m')

            best_test_acc = max(test_acc, best_test_acc)
            save_model(model, args, 'test', args.trial)

        # remember best acc@1 and save checkpoint
        if test_unseen_acc > best_test_unseen_acc:
            print('\033[92m' + 'Test Unseen Current best acc' + '\033[0m')

            best_test_unseen_acc = max(test_unseen_acc, best_test_unseen_acc)
            save_model(model, args, 'test-unseen', args.trial)

    print('Sources', args.source)
    print('Val best acc', best_val_acc)
    print('Test best acc', best_test_acc)
    print('Test best unseen acc', best_test_unseen_acc)

def data_loading(args):
    if args.data == 'PACS':
        num_domain = 3
        num_class = 7

        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_tranform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = datasets.__dict__['PACS']

        filter_class_list = [[0, 1, 3], [4, 0, 2], [5, 1, 2], [0,1,2,3,4,5]]

    if args.data == 'ImageNet':
        num_domain = 5
        num_class = 1000

        train_transform = transforms.Compose([
            #transforms.Resize(224),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_tranform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = datasets.__dict__['ImageNet']

        filter_class_list = [[],[],[],[],[],[]]

    if args.data == 'AWA2':
        num_domain = 5
        num_class = 50

        train_transform = transforms.Compose([
            #transforms.Resize(224),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_tranform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = datasets.__dict__['AWA2']

        filter_class_list = [None, None, None, None, True]

    train_source_iter_list = []
    train_source_iter_list_balanced = []
    for j, the_source in enumerate(args.source):
        train_source_dataset = dataset(root=args.data_root, domain=the_source, filter_class= filter_class_list[j],
                                       split='train', transform=train_transform, args=args)

        train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
        train_source_iter = ForeverDataIterator(train_source_loader)
        train_source_iter_list.append(train_source_iter)

        ## balanced sampler
        labels = train_source_dataset.label
        class_list = class_counter(labels, args.n_class)
        class_list[class_list==0] = 1.
        weight = 1./ class_list
        samples_weight = np.array([weight[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        dataloader_balance = DataLoader(train_source_dataset, batch_size=args.batch_size, num_workers=20,
                                             sampler=sampler, pin_memory=False)
        dataloader_balance_iter = ForeverDataIterator(dataloader_balance)
        train_source_iter_list_balanced.append(dataloader_balance_iter)

    val_loader = []
    for j, the_source in enumerate(args.source):
        val_source_dataset = dataset(root=args.data_root, domain=the_source, filter_class=filter_class_list[j],
                                       split='val', transform=val_tranform, args=args)

        val_source_loader = DataLoader(val_source_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.workers)
        val_loader.append(val_source_loader)

    test_loader = []
    for j, the_target in enumerate(args.target):
        if args.data == 'PACS':
            test_dataset = dataset(root=args.data_root, domain=the_target, filter_class=filter_class_list[-1],
                                           split='test', transform=val_tranform, args=args)
        else:
            test_dataset = dataset(root=args.data_root, domain=the_target, filter_class=args.source,
                                           split='test', transform=val_tranform, args=args)
        test_target_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers)
        test_loader.append(test_target_loader)

    return train_source_iter_list, train_source_iter_list_balanced, val_loader, test_loader

## model preparation
def model_loading(args):
    ## model

    model = models.__dict__['sfsc_classifier_embed_lt_meta'](args, data=args.data, backbone='resnet10', classifier='fc', embed=args.embed).cuda()

    ## optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                        momentum=0.9, weight_decay=1e-4, nesterov=False)
    ## scheduler
    if args.scheduler == 'stepLR':
        step_size = int(args.epochs * 0.4 * args.iters_per_epoch)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1) ## update per iter not per epoch

    return model, optimizer, lr_scheduler

def train_warm(train_source_iter_list, train_source_balanced_iter_list, model, loss_f, optimizer,
          lr_scheduler, epoch, args):
    ## metric meter
    cls_losses = AverageMeter('C Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    metric_losses = AverageMeter('Metric Loss', ':3.2f')
    aug_losses = AverageMeter('Aug Loss', ':3.2f')

    mapping_losses = AverageMeter('Cross Proto Loss', ':3.2f')
    meta_losses = AverageMeter('Meta C Loss', ':3.2f')
    meta_aug_losses = AverageMeter('Meta Aug Loss', ':3.2f')
    # mixup_losses = AverageMeter('Mixup Loss', ':3.2f')
    # # proto_losses = AverageMeter('Proto Loss', ':3.2f')
    # meta_proto_losses = AverageMeter('Meta Proto Loss', ':3.2f')
    # meta_cycle_losses = AverageMeter('Meta Cycle Loss', ':3.2f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [cls_losses, metric_losses, mapping_losses, meta_losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    tgt_all = []
    d_all = []
    f_all = []

    for i in range(args.iters_per_epoch):
        x_s_all = []
        y_s_all = []
        tgt_s_all = []
        f_s_all = []
        y_s_all_2 = []

        CV_Domain.freeze()

        domain_split = torch.randperm(args.n_domain-1)

        model.zero_grad()

        # -----------------  Meta Train  -----------------------------------
        fast_parameters = list(model.parameters())
        for weight in model.parameters():
            weight.fast = None
        model.zero_grad()

        cls_loss = []
        for domain in domain_split[:-1]:
            x_s, tgt_s, _ = next(train_source_iter_list[domain])
            x_s = x_s.cuda()
            tgt_s = tgt_s.cuda()

            y_s, f_s = model(x_s)
            cls_loss.append(loss_f[domain](y_s, tgt_s))

            x_s_all.append(x_s)
            tgt_s_all.append(tgt_s)
            y_s_all.append(y_s)
            f_s_all.append(f_s)
            # CV_Domain.update_statistics(f_s, tgt_s, torch.ones_like(tgt_s) * domain)

            tgt_all.append(tgt_s.cpu().data.numpy())
            f_all.append(f_s.cpu().data.numpy())
            d_all.append(domain * torch.ones_like(tgt_s).cpu().data.numpy())

        x_s_all = torch.cat(x_s_all, dim=0)
        y_s_all = torch.cat(y_s_all, dim=0)
        f_s_all = torch.cat(f_s_all, dim=0)
        tgt_s_all = torch.cat(tgt_s_all, dim=0)

        # cls_loss = loss_f(y_s_all, tgt_s_all) #+ 0.1*F.cross_entropy(y_s_all_2, tgt_s_all)
        cls_loss = torch.mean(torch.stack(cls_loss))
        metric_loss = torch.FloatTensor([0.]).cuda()#/= len(domain_split)    #, _ = model.mapping(f_s_all, tgt_s_all) # embed loss # = torch.FloatTensor([0.]).cuda()#
        metric_loss2 = torch.FloatTensor([0.]).cuda() #model.cross_prototype(domain_split, CV_Domain) #mapping loss #
        # if epoch<3:
        #     w = 0.1
        # elif epoch<6:
        #     w = 0.5
        # else:
        #     w = 1.
        w = args.w_metric
        meta_train_loss = cls_loss + (w*metric_loss + w*metric_loss2) #+ 0.5*mixup_feature_loss


        ## update metric
        cls_acc = accuracy(y_s_all, tgt_s_all)[0]
        cls_accs.update(cls_acc.item(), x_s_all.size(0))
        cls_losses.update(cls_loss.item(), x_s_all.size(0))
        metric_losses.update(metric_loss.item(), x_s_all.size(0))
        mapping_losses.update(metric_loss2.item(), x_s_all.size(0))

        optimizer.zero_grad()
        total_loss = meta_train_loss
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if i % args.print_freq == 0:
            progress.display(i)

    tgt_all = np.concatenate(tgt_all, axis=0)
    f_all = np.concatenate(f_all, axis=0)
    d_all = np.concatenate(d_all, axis=0)

    if args.vis and (epoch % 5 == 0):
        # plot_tsne(f_all, tgt_all, d_all, color='label', title='label',
        #           filename='tmp_results/' + mode + '_label_' + str(epoch))
        # plot_tsne(f_all, tgt_all, d_all, color='domain', title='domain',
        #           filename='tmp_results/' + mode + '_domain' + str(epoch))
        plot_dual_tsne(f_all, tgt_all, d_all, color='label', title='label', skip=5,
                       filename='tmp_results/' + 'train' + '_' + str(epoch))

## training
def train(train_source_iter_list, train_source_balanced_iter_list, model, loss_f, loss_aug_f, optimizer,
          lr_scheduler, epoch, args):
    ## metric meter
    cls_losses = AverageMeter('C Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    mapping_losses = AverageMeter('Mapping Loss', ':3.2f')
    crossproto_losses = AverageMeter('Cross Proto Loss', ':3.2f')
    aug_losses = AverageMeter('Aug Loss', ':3.2f')

    meta_mapping_losses = AverageMeter('Meta Mapping Loss', ':3.2f')
    meta_crossproto_losses = AverageMeter('Meta Cross Loss', ':3.2f')
    cycle_losses = AverageMeter('Cycle Loss', ':3.2f')
    meta_cls_losses = AverageMeter('Meta C Loss', ':3.2f')
    meta_aug_losses = AverageMeter('Meta Aug Loss', ':3.2f')

    # mixup_losses = AverageMeter('Mixup Loss', ':3.2f')
    # # proto_losses = AverageMeter('Proto Loss', ':3.2f')
    # meta_proto_losses = AverageMeter('Meta Proto Loss', ':3.2f')
    # meta_cycle_losses = AverageMeter('Meta Cycle Loss', ':3.2f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [cls_losses, mapping_losses, crossproto_losses, cycle_losses, aug_losses,
         meta_cls_losses, meta_mapping_losses, meta_crossproto_losses, meta_aug_losses,
         cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    tgt_all = []
    d_all = []
    f_all = []

    if epoch>=80:
        args.meta_step_size = 0.002
    elif epoch>=40:
        args.meta_step_size = 0.02

    for i in range(args.iters_per_epoch):
        x_s_all = []
        y_s_all = []
        tgt_s_all = []
        f_s_all = []
        y_s_all_2 = []

        CV_Domain.freeze()

        domain_split = torch.randperm(args.n_domain-1)

        model.zero_grad()

        # -----------------  Meta Train  -----------------------------------
        fast_parameters = list(model.parameters())
        for weight in model.parameters():
            weight.fast = None
        model.zero_grad()

        cls_loss = []
        metric_loss = []
        aug_loss = []
        for domain in domain_split[:-1]:
            x_s, tgt_s, _ = next(train_source_iter_list[domain])
            x_s = x_s.cuda()
            tgt_s = tgt_s.cuda()

            y_s, f_s = model(x_s)
            cls_loss.append(loss_f[domain](y_s, tgt_s))
            metric_loss.append(model.mapping(f_s, tgt_s, loss_f[domain]))
            if epoch>=args.epoch1:
                loss_aug_f.estimator.update_CV(f_s.detach(), tgt_s)
                aug_loss.append(loss_aug_f(model, f_s, y_s, tgt_s, args.lam*(epoch-args.epoch1)/(args.epochs-args.epoch1), loss_f[domain]))
            else:
                aug_loss.append(torch.FloatTensor([0.]).cuda())

            x_s_all.append(x_s)
            tgt_s_all.append(tgt_s)
            y_s_all.append(y_s)
            f_s_all.append(f_s)
            CV_Domain.update_statistics(f_s, tgt_s, torch.ones_like(tgt_s) * domain)

            tgt_all.append(tgt_s.cpu().data.numpy())
            f_all.append(f_s.cpu().data.numpy())
            d_all.append(domain * torch.ones_like(tgt_s).cpu().data.numpy())

        x_s_all = torch.cat(x_s_all, dim=0)
        y_s_all = torch.cat(y_s_all, dim=0)
        f_s_all = torch.cat(f_s_all, dim=0)
        tgt_s_all = torch.cat(tgt_s_all, dim=0)

        cls_loss = torch.mean(torch.stack(cls_loss)) # loss_f(y_s_all, tgt_s_all) #+ 0.1*F.cross_entropy(y_s_all_2, tgt_s_all)
        mapping_loss = torch.mean(torch.stack(metric_loss))
        aug_loss = torch.mean(torch.stack(aug_loss))
        cycle_loss = model.cycle_embed_loss()
        cross_proto_loss = model.cross_prototype(domain_split, CV_Domain)

        #metric_loss = torch.FloatTensor([0.]).cuda() #/= len(domain_split)    #, _ = model.mapping(f_s_all, tgt_s_all) # embed loss # = torch.FloatTensor([0.]).cuda()#
        #metric_loss2 = torch.FloatTensor([0.]).cuda() #model.cross_prototype(domain_split, CV_Domain) #mapping loss #
        # if epoch<3:
        #     w = 0.1
        # elif epoch<6:
        #     w = 0.5
        # else:
        #     w = 1.
        w_map = 0.1
        w_cycle = 0.1
        w_cross = 0.1
        w_aug = 0.5
        meta_train_loss = cls_loss + w_map * mapping_loss + w_cycle * cycle_loss + w_cross * cross_proto_loss + w_aug * aug_loss

        #+0.5*mixup_feature_loss

        # ----------------------- Meta Test -----------------------------------
        grad = torch.autograd.grad(meta_train_loss, fast_parameters,
                                   create_graph=True, allow_unused=True)
        if args.stop_gradient:
            grad = [g.detach() for g in
                    grad]

        fast_parameters = []

        torch.nn.utils.clip_grad_norm_(grad, 2.)

        for k, weight in enumerate(model.parameters()):
            if weight.fast is None: # this is to enable fast parameter
                weight.fast = weight - args.meta_step_size * grad[k]
            else:
                weight.fast = weight.fast - args.meta_step_size * grad[
                    k]
            fast_parameters.append(
                weight.fast)

        x_s, tgt_s, _ = next(train_source_iter_list[domain_split[-1]])
        x_s = x_s.cuda()
        tgt_s = tgt_s.cuda()
        y_s, f_s = model(x_s)
        meta_class_loss = loss_f[domain_split[-1]](y_s, tgt_s)
        meta_mapping_loss = model.mapping(f_s, tgt_s, loss_f[domain_split[-1]])
        meta_cross_loss = model.cross_domain_prototype(f_s, domain_split[-1], tgt_s, CV_Domain, loss_f[domain_split[-1]])
        #meta_cross_loss =
        if epoch >= args.epoch1:
            meta_aug_loss = loss_aug_f(model, f_s, y_s, tgt_s, args.lam*(epoch-args.epoch1)/(args.epochs-args.epoch1), loss_f[domain_split[-1]])
        else:
            meta_aug_loss = torch.FloatTensor([0.]).cuda()
        meta_val_loss = meta_class_loss + w_map*meta_mapping_loss + w_cross*meta_cross_loss + w_aug * meta_aug_loss

        ## update metric
        cls_acc = accuracy(y_s_all, tgt_s_all)[0]
        cls_accs.update(cls_acc.item(), x_s_all.size(0))
        cls_losses.update(cls_loss.item(), x_s_all.size(0))

        mapping_losses.update(mapping_loss.item(), x_s_all.size(0))
        crossproto_losses.update(cross_proto_loss.item(), x_s_all.size(0))
        aug_losses.update(aug_loss.item(), x_s_all.size(0))

        meta_cls_losses.update(meta_class_loss.item(), x_s_all.size(0))
        meta_mapping_losses.update(meta_mapping_loss.item(), x_s_all.size(0))
        meta_crossproto_losses.update(meta_cross_loss.item(), x_s_all.size(0))
        cycle_losses.update(cycle_loss.item(), x_s_all.size(0))
        meta_aug_losses.update(meta_aug_loss.item(), x_s_all.size(0))

        optimizer.zero_grad()
        total_loss = meta_train_loss + args.w_val*meta_val_loss
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if i % args.print_freq == 0:
            progress.display(i)

    tgt_all = np.concatenate(tgt_all, axis=0)
    f_all = np.concatenate(f_all, axis=0)
    d_all = np.concatenate(d_all, axis=0)

    if args.vis and (epoch % 5 == 0):
        # plot_tsne(f_all, tgt_all, d_all, color='label', title='label',
        #           filename='tmp_results/' + mode + '_label_' + str(epoch))
        # plot_tsne(f_all, tgt_all, d_all, color='domain', title='domain',
        #           filename='tmp_results/' + mode + '_domain' + str(epoch))
        plot_dual_tsne(f_all, tgt_all, d_all, color='label', title='label', skip=5,
                       filename='tmp_results/' + 'train' + '_' + str(epoch))


## validation
def val(val_loader_list, model, epoch, mode, best_score, args):
    ## metric meter
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    metric_meter = MeanTopKRecallMeter_domain(num_classes=args.n_class, num_domain=len(val_loader_list))

    f_all = []
    d_all = []
    x_all = []
    y_all = []
    tgt_all = []

    for weight in model.parameters():
        weight.fast = None

    model.eval()
    for domain, val_loader in enumerate(val_loader_list):
        for i, (x_s, tgt_s, _) in enumerate(val_loader):
            x_s = x_s.cuda()
            tgt_s = tgt_s.cuda()

            ## compute output
            with torch.no_grad():
                y_s, f_s = model(x_s)

            if args.vis:
                x_all.append(x_s.cpu().data.numpy())
                y_all.append(y_s.cpu().data.numpy())
                tgt_all.append(tgt_s.cpu().data.numpy())
                f_all.append(f_s.cpu().data.numpy())
                d_all.append(domain*torch.ones_like(tgt_s).cpu().data.numpy())

            metric_meter.add(y_s.cpu().data.numpy(),
                            tgt_s.cpu().data.numpy(),
                            domain*torch.ones_like(tgt_s).cpu().data.numpy())

        # cls_acc = accuracy(y_s_all, tgt_s_all)[0]

        print('domain %d acc %.3f.' % (domain, metric_meter.domain_aware_acc(domain)))

    #x_all = np.concatenate(x_all, axis=0)
    #y_all = np.concatenate(y_all, axis=0)


    #prototype = model.proj_s2v(model.embed)
    #print(prototype[tgt_all].shape)
    #print(f_all.shape)
    # print(F.mse_loss(prototype[tgt_all], torch.FloatTensor(f_all).cuda()))
    # print(f_all)
    #prototype = prototype.cpu().data.numpy()

    if args.vis and (epoch % 5 == 0):
        tgt_all = np.concatenate(tgt_all, axis=0)
        f_all = np.concatenate(f_all, axis=0)
        d_all = np.concatenate(d_all, axis=0)
        # plot_tsne(f_all, tgt_all, d_all, color='label', title='label',
        #           filename='tmp_results/' + mode + '_label_' + str(epoch))
        # plot_tsne(f_all, tgt_all, d_all, color='domain', title='domain',
        #           filename='tmp_results/' + mode + '_domain' + str(epoch))
        plot_dual_tsne(f_all, tgt_all, d_all, color='label', title='label',
                             filename='tmp_results/' + mode + '_'+ str(epoch))

        # plot_tsne_prototype(f_all, tgt_all, d_all, prototype, color='label', title='label',
        #                      filename='tmp_results/' + mode + '_proto_'+ str(epoch))

        # plot_tsne_prototype(f_all/np.linalg.norm(f_all), tgt_all, d_all, prototype/np.linalg.norm(prototype),
        #                     color='label', title='label',
        #                      filename='tmp_results/' + mode + '_proto_norm_'+ str(epoch))
    # print(all_acc_)
    print('Mean %s acc %.3f. [%.3f]' % (mode, metric_meter.domain_mean_acc(), max(metric_meter.domain_mean_acc(), best_score)))
    if args.print_flag:
        metric_meter.domain_class_aware_recall()

    return metric_meter.domain_mean_acc(), metric_meter.domain_aware_acc(domain)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configurations for LTDG')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('-d', '--data', metavar='DATA', default='ImageNet')
    parser.add_argument('-s', '--source', type=str, default='SHUV', help='source domain(s)')
    parser.add_argument('-t', '--target', type=str, default='SHUVO', help='target domain(s)')
    parser.add_argument('--data_root', type=str, default='/home/dev/Data/xiao/data/ImageNet_LT/', help='data root')
    parser.add_argument('--lt', action='store_true')
    parser.add_argument('--gpu', default='1', type=str, help='gpu id ')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=24, type=int,
                        metavar='N',
                        help='mini-batch size (default: 24)')
    parser.add_argument('-i', '--iters_per_epoch', default=101, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--scheduler', type=str, default='stepLR')
    parser.add_argument('--lr', default=0.2, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--vis', type=int, default=0)
    parser.add_argument('--trial', type=str, default='test_lt_aug', help='current trial name for checkpoint save')
    parser.add_argument("--stop_gradient", type=int, default=1,
                        help='whether stop gradient of the first order gradient')
    parser.add_argument("--print_flag", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=30.)
    parser.add_argument('--loss', type=str, default='BSCE')
    parser.add_argument('--embed', type=str, default='wiki')
    parser.add_argument('--w_metric', type=float, default=0.1)
    parser.add_argument("--meta_step_size", type=float, default=0.2)
    parser.add_argument("--w_val", type=float, default=0.3)
    parser.add_argument("--epoch1", type=int, default=40)
    parser.add_argument("--lam", type=float, default=10.)


    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.data == 'PACS':
        args.iters_per_epoch = 101
        args.n_class = 7
        args.batch_size = 24
        args.n_domain = 4
    if args.data == 'ImageNet':
        args.iters_per_epoch = 200
        args.n_class = 1000
        args.batch_size = 128
        args.epochs = 100
        args.n_domain = 5
    if args.data == 'AWA2':
        args.n_class = 50
        args.batch_size = 48
        args.iters_per_epoch = 50
        args.epochs = 100
        args.n_domain = 5

    print(args)

    CV_Domain = Domain_Alignment(512, args.n_class, args.n_domain-1)

    main(args)




