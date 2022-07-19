import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data import ltds_dataset
from .tools import ForeverDataIterator

def get_data(cfg):
    """
    Getting datasets for AWA2-LTS and ImageNet-LTS
    :param cfg:
    :return:
    """
    train_transform = transforms.Compose([
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

    dataset = cfg.DATA.NAME
    dataset_path = cfg.DATA.ROOT + '/' + dataset + '-lts'
    source = cfg.DATA.SOURCE
    target = cfg.DATA.TARGET
    bs = cfg.DATA.BATCH_SIZE
    n_worker = cfg.DATA.NUM_WORKERS

    # training sources
    train_source_iter_list = []
    for j, the_source in enumerate(source):
        train_source_dataset = ltds_dataset(root=dataset_path, domain=the_source,
                                       split='train', transform=train_transform,)
        train_source_loader = DataLoader(train_source_dataset, batch_size=bs,
                                         shuffle=True, num_workers=n_worker, drop_last=True)
        train_source_iter = ForeverDataIterator(train_source_loader)
        train_source_iter_list.append(train_source_iter)

    val_loader = []
    for j, the_source in enumerate(source):
        val_source_dataset = ltds_dataset(root=dataset_path, domain=the_source,
                                     split='val', transform=val_tranform)
        val_source_loader = DataLoader(val_source_dataset, batch_size=bs,
                                       shuffle=False, num_workers=n_worker)
        val_loader.append(val_source_loader)

    test_loader = []
    for j, the_target in enumerate(target):
        test_dataset = ltds_dataset(root=dataset_path, domain=the_target,
                                   split='test', transform=val_tranform,)
        test_target_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False,
                                        num_workers=n_worker)
        test_loader.append(test_target_loader)

    return train_source_iter_list, val_loader, test_loader

if __name__ == '__main__':
    from config import config_parser, cfg, update_config

    args = config_parser.parse_args()
    update_config(cfg, args)

    get_data(cfg)
