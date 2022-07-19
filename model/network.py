import os
from config import config_parser, cfg, update_config

args = config_parser.parse_args()
update_config(cfg, args)
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from .basics import ResNetFast, BasicBlock_fw, BatchNorm1d_fw, Linear_fw, ResNet, BasicBlock

import scipy.io as sio
from itertools import combinations


class sfsc_meta(nn.Module):
    """
    The dataset used for data common settings as well as those kinds of rese
    """
    def __init__(self, cfg):
        super(sfsc_meta, self).__init__()

        dataset = cfg.DATA.NAME
        net_f = cfg.MODEL.F
        net_c = cfg.MODEL.C
        embed = cfg.EMBED.NAME
        pretrained_flag = cfg.MODEL.PRETRAIN_FLAG

        ## backbone model net_f
        if net_f == 'resnet18':
            model = ResNetFast(BasicBlock_fw, [2, 2, 2, 2])
            self.dim_f = 512

            if pretrained_flag:
                pretrained = models.resnet18(pretrained=pretrained_flag)
                model.load_state_dict(pretrained.state_dict(), strict=False)

            del model.fc
            self.net_f = model
            del model

        if net_f == 'resnet10':
            model = ResNetFast(BasicBlock_fw, [1, 1, 1, 1])
            self.dim_f = 512

            del model.fc
            self.net_f = model
            del model

        ## classifier
        if net_c == 'fc':
            self.net_c = Linear_fw(self.dim_f, cfg.DATA.CLASS)

        ## projector
        s = sio.loadmat(cfg.DATA.ROOT + dataset + '-lts/embedding/' + dataset + '_' + embed + '.mat')['latent']
        s = torch.FloatTensor(s).cuda()
        self.s = F.normalize(s)
        self.dim_e = self.s.shape[1]

        self.proj_e = nn.Sequential(Linear_fw(self.dim_f, self.dim_e),
                                    BatchNorm1d_fw(self.dim_e),
                                    nn.ReLU()) # v2s
        self.proj_d = nn.Sequential(Linear_fw(self.dim_e, self.dim_f),
                                    BatchNorm1d_fw(self.dim_f),
                                    nn.ReLU()) # s2v

        self.margin = cfg.EMBED.MARGIN
        self.scale = cfg.EMBED.SCALE

    def get_parameters(self):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.proj_e.parameters(), "lr_mult": 1.0},
            {"params": self.proj_d.parameters(), "lr_mult": 1.0},
            {"params": self.net_f.parameters(), "lr_mult": 1.0},
            {"params": self.net_c.parameters(), "lr_mult": 1.0},
        ]
        return params

    def forward(self, x):
        """forward function"""
        f = self.net_f(x)
        f = f.reshape(f.shape[0], -1)
        pred = self.net_c(f)

        return pred, f

    def mapping_v2s(self, f, target, loss_f):
        """mapping from visual to semantic space"""
        margin = self.margin
        scale = self.scale

        onehot = F.one_hot(target, self.s.shape[0])

        f_s = self.proj_e(f)
        s = self.s

        f_s_norm = F.normalize(f_s)
        score_v2s = f_s_norm @ s.T
        psi_score_v2s = score_v2s - margin

        contrast_score_v2s = scale * torch.where(onehot==1, psi_score_v2s, score_v2s)

        return loss_f(contrast_score_v2s, target)

    def mapping_s2s(self, domains, CV):
        """align prototypes across domains"""
        margin = self.margin
        scale = self.scale

        tgt = torch.arange(self.s.shape[0]).cuda()
        n_c = self.s.shape[0]
        onehot = F.one_hot(tgt, n_c*2-1)

        loss_s2s = 0.
        counter = 0.

        for (i, j) in combinations(domains, 2):
            proto_i = CV.mean[i]
            proto_i_v2s = torch.zeros_like(self.s)
            proto_i_v2s[CV.Amount[i] != 0] = self.proj_e(proto_i[CV.Amount[i] != 0])
            proto_i_v2s[CV.Amount[i] == 0] = self.s[CV.Amount[i] == 0]
            proto_i_v2s_norm = F.normalize(proto_i_v2s)  # / pro_i.norm(keepdim=True, dim=-1)

            proto_j = CV.mean[j]
            proto_j_v2s = torch.zeros_like(self.s)
            proto_j_v2s[CV.Amount[j] != 0] = self.proj_e(proto_j[CV.Amount[j] != 0])
            proto_j_v2s[CV.Amount[j] == 0] = self.s[CV.Amount[j] == 0]
            proto_j_v2s_norm = F.normalize(proto_j_v2s)  # / pro_j.norm(keepdim=True, dim=-1)

            score_i2i = proto_i_v2s_norm @ proto_j_v2s_norm.T
            score_i2i = torch.masked_select(score_i2i, ~torch.eye(n_c, dtype=bool).cuda()).reshape(n_c, n_c - 1)

            score_j2j = proto_j_v2s_norm @ proto_j_v2s_norm.T
            score_j2j = torch.masked_select(score_j2j, ~torch.eye(n_c, dtype=bool).cuda()).reshape(n_c, n_c - 1)

            score_i2j = proto_i_v2s_norm @ proto_j_v2s_norm.T
            score_i2j = torch.cat((score_i2j, score_i2i), dim=1)

            score_j2i = proto_j_v2s_norm @ proto_i_v2s_norm.T
            score_j2i = torch.cat((score_j2i, score_j2j), dim=1)

            psi_score_i2j = score_i2j - margin
            psi_score_j2i = score_j2i - margin

            contrast_score_i2j = scale * torch.where(onehot == 1, psi_score_i2j, score_i2j)
            contrast_score_j2i = scale * torch.where(onehot == 1, psi_score_j2i, score_j2i)

            loss_s2s += 0.5 * F.cross_entropy(contrast_score_i2j, tgt) + 0.5 * F.cross_entropy(contrast_score_j2i, tgt)

            counter += 1

        return loss_s2s / counter

    def mapping_s2v(self, domains, CV):
        """mapping from semantic to visual space"""
        margin = self.margin
        scale = self.scale

        n_c = self.s.shape[0]

        loss_s2v = 0.
        counter = 0.

        for i in domains:
            proto_i = CV.mean[i]
            proto_i_v2s = torch.zeros_like(self.s)
            proto_i_v2s[CV.Amount[i] != 0] = self.proj_e(proto_i[CV.Amount[i] != 0])
            proto_i_v2s[CV.Amount[i] == 0] = self.s[CV.Amount[i] == 0]
            proto_i_v2s_norm = F.normalize(proto_i_v2s)  # / pro_i.norm(keepdim=True, dim=-1)

            proto_i_v2s_s2v = self.proj_d(proto_i_v2s_norm)
            contrast_score_s2v = self.net_c(proto_i_v2s_s2v)

            # cycle
            proto_i_v2s_s2v_v2s = self.proj_e(proto_i_v2s_s2v)
            proto_i_v2s_s2v_v2s_norm = F.normalize(proto_i_v2s_s2v_v2s)  # / embed_s2v_v2s.norm(keepdim=True, dim=-1)
            score_cycle = proto_i_v2s_s2v_v2s_norm @ self.s.T
            psi_score_cycle = score_cycle - margin

            target_cycle = torch.arange(0, n_c).cuda()
            onehot_cycle = F.one_hot(target_cycle, n_c)
            contrast_score_cycle = scale * torch.where(onehot_cycle == 1, psi_score_cycle, score_cycle)

            counter += 1
            loss_s2v += 0.5*F.cross_entropy(contrast_score_s2v, target_cycle) + 0.5*F.cross_entropy(contrast_score_cycle, target_cycle)

        return loss_s2v/counter

    def cross_domain_prototype(self, f, d, y, CV, loss_f):
        """mapping from visual space to prototype semantic space of other domains"""
        # here we assume only one meta-test domain, and all the others
        # to make it more flexible to domain splits

        margin = self.margin
        scale = self.scale

        onehot = F.one_hot(y, self.s.shape[0])

        f_v2s = self.proj_e(f)
        f_v2s = F.normalize(f_v2s)  # / f_v2s.norm(keepdim=True, dim=-1)

        loss = 0.

        ## cross domain
        for domain in range(CV.n_domain):
            if domain == d:
                continue
            pro = CV.mean[domain].detach()
            pro_v2s = torch.zeros_like(self.s)
            pro_v2s[CV.Amount[domain] != 0] = self.proj_e(pro[CV.Amount[domain] != 0])
            pro_v2s[CV.Amount[domain] == 0] = self.s[CV.Amount[domain] == 0]
            pro_v2s_norm = F.normalize(pro_v2s)

            score_v2s = f_v2s @ pro_v2s_norm.T
            psi_score_v2s = score_v2s - margin
            contrast_score_v2s = scale * torch.where(onehot == 1, psi_score_v2s, score_v2s)

            loss += loss_f(contrast_score_v2s, y)

        return loss/(CV.n_domain-1)


class sfsc(sfsc_meta):
    def __init__(self, cfg):
        super(sfsc, self).__init__(cfg)

        dataset = cfg.DATA.NAME
        net_f = cfg.MODEL.F
        net_c = cfg.MODEL.C
        embed = cfg.EMBED.NAME
        pretrained_flag = cfg.MODEL.PRETRAIN_FLAG

        ## backbone model net_f
        if net_f == 'resnet18':
            model = ResNet(BasicBlock, [2, 2, 2, 2])
            self.dim_f = 512

            if pretrained_flag:
                pretrained = models.resnet18(pretrained=pretrained_flag)
                model.load_state_dict(pretrained.state_dict(), strict=False)

            del model.fc
            self.net_f = model
            del model

        if net_f == 'resnet10':
            model = ResNet(BasicBlock, [1, 1, 1, 1])
            self.dim_f = 512

            del model.fc
            self.net_f = model
            del model

        ## classifier
        if net_c == 'fc':
            self.net_c = nn.Linear(self.dim_f, cfg.DATA.CLASS)

        ## projector
        s = sio.loadmat('dataset/' + dataset + '_' + embed + '.mat')['latent']
        s = torch.FloatTensor(s).cuda()
        self.s = F.normalize(s)
        self.dim_e = self.s.shape[1]

        self.proj_e = nn.Sequential(nn.Linear(self.dim_f, self.dim_e),
                                    nn.BatchNorm1d(self.dim_e),
                                    nn.ReLU()) # v2s
        self.proj_d = nn.Sequential(nn.Linear(self.dim_e, self.dim_f),
                                    nn.BatchNorm1d(self.dim_f),
                                    nn.ReLU()) # s2v

        self.margin = cfg.EMBED.MARGIN
        self.scale = cfg.EMBED.SCALE

if __name__ == '__main__':
    model = sfsc_meta(cfg)
