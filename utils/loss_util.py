import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .tools import class_counter

def get_loss(labels, num, cfg=None):
    """
    get loss perdomain
    :param labels: set of labels of each training domain
    :param num: class num in total
    :param cfg: LOSS.PER_DOMAIN: for those class weight specific loss, whether aggreate or treat each domain seperately
    :return:
    """

    if not cfg.LOSS.PER_DOMAIN_FLAG:
        num_domain = len(labels)
        labels_agg = np.concatenate(labels)
        labels = [labels_agg] * num_domain

    loss_f = []
    for i in range(len(labels)):
        if cfg.LOSS.TYPE == 'CE':
            loss_f.append(CE(labels[i], num))

        if cfg.LOSS.TYPE == 'BSCE':
            loss_f.append(BSCE(labels[i], num))

        if cfg.LOSS.TYPE == 'SEQL':
            loss_f.append(SEQL(labels[i], num, cfg))

    return loss_f

class CE(nn.Module):
    def __init__(self, labels, num):
        super(CE, self).__init__()
        self.num = num
        self.num_cls_list = class_counter(labels, num)
        self.weight_list = None

    def forward(self, pred, label, **kwargs):
        loss = F.cross_entropy(pred, label)

        return loss

class BSCE(CE):
    def __init__(self, labels, num):
        super(BSCE, self).__init__(labels, num)
        self.bsce_weight = torch.FloatTensor(self.num_cls_list).cuda()

    def forward(self, pred, label, **kwargs):
        logits = pred + self.bsce_weight.unsqueeze(0).expand(pred.shape[0], -1).log()
        loss = F.cross_entropy(logits, label)

        return loss

class SEQL(CE):
    def __init__(self, labels, num, cfg=None):
        super(SEQL, self).__init__(labels, num)
        self.lambda_ = cfg.LOSS.SEQL.LAMBDA
        self.ignore_prob = cfg.LOSS.SEQL.IGNORE
        self.class_weight = torch.FloatTensor(self.num_cls_list > self.lambda_*len(labels)).cuda()

    def replace_masked_values(self, tensor, mask, replace_with):
        assert tensor.dim() == mask.dim(), '{} vs {}'.format(tensor.shape, mask.shape)
        one_minus_mask = 1 - mask
        values_to_add = replace_with * one_minus_mask
        return tensor * mask + values_to_add

    def forward(self, pred, label, **kwargs):
        N, C = pred.shape
        not_ignored = self.class_weight.view(1, C).repeat(N, 1)
        over_prob = (torch.rand(pred.shape).cuda() > self.ignore_prob).float()
        is_gt = label.new_zeros((N, C)).float()
        is_gt[torch.arange(N), label] = 1

        weights = ((not_ignored + over_prob + is_gt) > 0).float()
        input = self.replace_masked_values(pred, weights, -1e7)
        loss = F.cross_entropy(input, label)
        return loss


class EstimatorCV():
    def __init__(self, feature_num, class_num, embed):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.ones(class_num).cuda()
        self.embed = F.normalize(embed, dim=-1)

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)

        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1).expand(C, A)
        )

        weight_CV[weight_CV != weight_CV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul((self.Ave - ave_CxA).pow(2))

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                           .mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_CV) + ave_CxA.mul(weight_CV)).detach()

        self.Amount += onehot.sum(0)

    def new_covariance(self, k):
        similarity = self.embed @ self.embed.T
        topk_index = torch.topk(similarity, k, -1)[1]
        new_Cov = []
        new_Amount = []
        for i in range(k): # include self ?
            new_Cov.append(self.CoVariance[topk_index[:, i]])
            new_Amount.append(self.Amount[topk_index[:, i]])

        new_Cov = torch.stack(new_Cov) # k*c*f
        new_Amount = torch.stack(new_Amount) # k*c
        new_Amount /= torch.sum(new_Amount, dim=0, keepdim=True)
        new_Cov = torch.sum(new_Cov * new_Amount.view(k, new_Amount.shape[1], 1), dim=0)

        return new_Cov


class ISDALoss(nn.Module):
    """Implicit Semantic Data Augmentation"""
    def __init__(self, feature_num, class_num, embed):
        super(ISDALoss, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num, embed)
        self.class_num = class_num
        self.cross_entropy = nn.CrossEntropyLoss()
        # self.embed = embed

    def isda_aug(self, fc, features, y, labels, cv_matrix, ratio):

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0]
        #print(weight_m.shape)

        if weight_m.fast is None:
            NxW_ij = weight_m.reshape(1, C, A).expand(N, C, A)
        else:
            NxW_ij = weight_m.fast.reshape(1, C, A).expand(N, C, A)

        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.reshape(N, 1, 1)
                              .expand(N, C, A))

        CV_temp = cv_matrix[labels]

        sigma2 = ratio * (weight_m - NxW_kj).pow(2).mul(
            CV_temp.view(N, 1, A).expand(N, C, A)
        ).sum(2)

        aug_result = y + 0.5 * sigma2

        return aug_result

    def forward(self, model, features, y, target_x, ratio, loss_f):
        Cov = self.estimator.new_covariance(5)

        isda_aug_y = self.isda_aug(model.net_c, features, y, target_x, Cov.detach(), ratio)
        #print(isda_aug_y.shape)

        loss = loss_f(isda_aug_y, target_x)

        return loss#, y

    def forward_noweight(self, model, features, y, target_x, ratio, loss_f):
        Cov = self.estimator.CoVariance

        isda_aug_y = self.isda_aug(model.net_c, features, y, target_x, Cov.detach(), ratio)
        loss = loss_f(isda_aug_y, target_x)

        return loss

class Prototype(torch.nn.Module):
    """Prototype Generation"""
    def __init__(self, feat_dim, n_class, n_domain):
        super(Prototype, self).__init__()
        self.n_domain = n_domain
        self.n_class = n_class

        self.mean = list()
        self.var = list()

        for i in range(n_domain):
            self.mean.append(torch.zeros(n_class, feat_dim).cuda())
            self.var.append(torch.zeros(n_class, feat_dim).cuda())

            self.Amount = torch.zeros(n_domain, n_class).cuda()

        self.beta = 0.5

    def update_statistics(self, feats, labels, domains, epsilon=1e-5):
        num_labels = 0

        ## moving average to calculate prototype
        for domain_idx in torch.unique(domains):
            tmp_feat = feats[domains == domain_idx]
            tmp_label = labels[domains == domain_idx]
            num_labels += tmp_label.shape[0]

            onehot_label = torch.zeros((tmp_label.shape[0], self.n_class)).scatter_(1, tmp_label.unsqueeze(
                -1).cpu(), 1).float().cuda()
            domain_feature = tmp_feat.unsqueeze(1) * onehot_label.unsqueeze(-1)
            tmp_mean = domain_feature.sum(0) / (onehot_label.unsqueeze(-1).sum(0) + epsilon)

            tmp_mask = (tmp_mean.sum(-1) != 0).float().unsqueeze(-1)

            self.mean[domain_idx] = self.mean[domain_idx].detach() * (1 - tmp_mask) + (
                    self.mean[domain_idx].detach() * self.beta + tmp_mean * (1 - self.beta)) * tmp_mask
            self.Amount[domain_idx][torch.unique(tmp_label)] = 1

    def freeze(self):
        for d in range(len(self.mean)):
            self.mean[d] = self.mean[d].detach()

if __name__ == '__main__':
    from config import config_parser, cfg, update_config
    from data_util import get_data

    args = config_parser.parse_args()
    update_config(cfg, args)

    train_data, val_data, test_data = get_data(cfg)

    label_set = []
    for dataloader in train_data:
        train_label = dataloader.data_loader.dataset.label
        label_set.append(train_label)

    loss_f = get_loss(label_set, cfg.DATA.CLASS, cfg)
