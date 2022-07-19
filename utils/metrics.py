import numpy as np
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class MeanTopKRecallMeter_domain(object):
    """Computes the Class-Mean TopK Recall Per Domain"""
    def __init__(self, num_classes, num_domain, k=1):
        self.num_classes = num_classes
        self.num_domain = num_domain
        self.k = k
        self.reset()

    def reset(self):
        self.tps = np.zeros((self.num_domain, self.num_classes))
        self.nums = np.zeros((self.num_domain, self.num_classes))

    def add(self, scores, labels, id):
        tp = (np.argsort(scores, axis=1)[:, -self.k:] == labels.reshape(-1, 1)).max(1)
        for s,l in np.unique(np.stack((id, labels),axis=1), axis=0):
            self.tps[s, l]+=tp[(id==s) & (labels==l)].sum()
            self.nums[s, l]+=((id==s) & (labels==l)).sum()

    def value(self):
        recalls = (self.tps.sum(axis=0)/self.nums.sum(axis=0))[self.nums.sum(axis=0)>0]
        if len(recalls)>0:
            return recalls.mean()*100
        else:
            return None

    def class_aware_recall(self):
        recalls = (self.tps.sum(axis=0)/self.nums.sum(axis=0))[self.nums.sum(axis=0)>0]
        print(("Class: {:.2f}, "*len(recalls)).format(*recalls))

    def domain_aware_recall(self):
        recalls = (self.tps.sum(axis=1) / self.nums.sum(axis=1))[self.nums.sum(axis=1) > 0]
        print(("Domain: {:.2f}, " *len(recalls)).format(*recalls))

    def domain_class_aware_recall(self):
        recalls = (self.tps / self.nums)
        print(np.round(recalls, 2))

    def domain_aware_acc(self, domain_idx):
        acc = self.tps[domain_idx].sum() / self.nums[domain_idx].sum()
        return acc

    def overall_acc(self):
        acc = self.tps.sum().sum() / self.nums.sum().sum()
        return acc

    def domain_mean_acc(self):
        accs = self.tps.sum(1) / self.nums.sum(1)
        return accs.mean()


def Accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class ProgressMeter(object):
    """Logging Progreess"""
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'