import torch
import model as models

def get_model(cfg):
    model_name = cfg.MODEL.NAME

    ## model
    model = models.__dict__[model_name](cfg).cuda()

    ## optimizer
    if cfg.TRAIN.OPTIMIZER.TYPE.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.get_parameters(), cfg.TRAIN.OPTIMIZER.BASELR,
                        momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM, weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=False)
    ## scheduler
    if cfg.TRAIN.LR_SCHEDULER.TYPE.lower() == 'steplr':
        step_size = int(cfg.TRAIN.MAX_EPOCH * 0.4 * cfg.TRAIN.ITER_PER_EPOCH)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    return model, optimizer, lr_scheduler


if __name__ == '__main__':
    from config import config_parser, cfg, update_config

    args = config_parser.parse_args()
    update_config(cfg, args)

    get_model(cfg)