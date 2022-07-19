'''
1. Put this script in the first layer of Cycle-GAN repo
'''

import glob
import os
import argparse

import torch
import torchvision.utils as vutils

#from options.test_options import TestOptions
from data.base_dataset import get_transform
from models import create_model

from PIL import Image
from tqdm import tqdm

from options.base_options import BaseOptions

class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')

        # Added LT-DS
        parser.add_argument('--dataset', default='awa2', choices=['awa2', 'imagenet'])
        parser.add_argument('--datapath', default='.')
        parser.add_argument('--style', default='vangogh', choices=['vangogh', 'ukiyoe'])

        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser

if __name__ == '__main__':
    opt = TestOptions().parse()

    ## preprocess transformation
    DATA = opt.datapath
    style = opt.style
    dataset = opt.dataset

    opt.batch_size = 1
    opt.num_threads = 0
    opt.name = 'style_' + style + '_pretrained'
    opt.model = 'test'
    opt.no_dropout = True
    opt.no_flip =True
    opt.preprocess = 'none'
    model = create_model(opt)
    model.setup(opt)

    img_transform = get_transform(opt)

    # transfer
    model.eval()
    if dataset == 'imagenet':

        filelist = glob.glob(DATA + '/imagenet-lts/original/*.JPEG')
        savedir = DATA + '/imagenet/' + style

        if not os.path.isdir(savedir):
            os.makedirs(savedir)

        for i, filename in tqdm(enumerate(filelist)):
            Img = Image.open(filename).convert('RGB')
            h = Img.size[0]
            w = Img.size[1]

            # Img.size()
            Img = img_transform(Img)
            Img = Img.view(1, Img.shape[0], Img.shape[1], Img.shape[2])
            with torch.no_grad():
                result = model.netG(Img)
            result = result[0]

            result = result[[0, 1, 2], :, :]
            result = result.data.cpu().float() * 0.5 + 0.5

            vutils.save_image(result, os.path.join(savedir, filename.split('/')[-1][:-5] + '_' + style + '.jpg'))

    if dataset == 'awa2':
        fold_list = glob.glob(DATA+'/awa2-lts/original/*')
        print(fold_list)
        for fold in fold_list:
                print(fold)
                filelist = glob.glob(fold + '/*jpg')
                savedir = fold.replace('original', opt.style.lower())

                if not os.path.isdir(savedir):
                        os.makedirs(savedir)

                for i, filename in tqdm(enumerate(filelist)):
                    Img = Image.open(filename).convert('RGB')
                    h = Img.size[0]
                    w = Img.size[1]

                    # Img.size()
                    Img = img_transform(Img)
                    Img = Img.view(1, Img.shape[0], Img.shape[1], Img.shape[2])
                    with torch.no_grad():
                        result = model.netG(Img)
                    result = result[0]

                    result = result[[0, 1, 2], :, :]
                    result = result.data.cpu().float() * 0.5 + 0.5

                    vutils.save_image(result,
                                      os.path.join(savedir, filename.split('/')[-1][:-4] + '_' + style + '.jpg'))

print('Done!')
