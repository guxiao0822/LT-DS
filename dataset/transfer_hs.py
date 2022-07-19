'''
1. Put this script in the first layer of CartoonGAN repo
'''

import torch
import os
import numpy as np
import argparse
from PIL import Image
import glob
from tqdm import tqdm

import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils

from network.Transformer import Transformer

parser = argparse.ArgumentParser()
parser.add_argument('--load_size', default=256)
parser.add_argument('--model_path', default='./pretrained_model')
parser.add_argument('--dataset', default='awa2', choices=['awa2', 'imagenet'])
parser.add_argument('--datapath', default='.')
parser.add_argument('--style', default='Hayao', choices=['Hayao', 'Shinkai'])
parser.add_argument('--gpu', type=int, default=0)

opt = parser.parse_args()

# valid_ext = ['.jpg', '.png', '.JPEG']

DATA = opt.datapath
style = opt.style
dataset = opt.dataset

# load pretrained model
model = Transformer()
model.load_state_dict(torch.load(os.path.join(opt.model_path, opt.style + '_net_G_float.pth')))
model.eval()

if opt.gpu > -1:
        print('GPU mode')
        model.cuda()
else:
        print('CPU mode')
        model.float()

# style transfer
if dataset == 'awa2':
        fold_list = glob.glob(DATA+'/awa2-lts/original/*')
        for fold in fold_list:
                print(fold)
                filelist = glob.glob(fold + '/*jpg')
                savedir = fold.replace('original', opt.style.lower())

                if not os.path.isdir(savedir):
                        os.makedirs(savedir)

                for files in tqdm(filelist):
                        # ext = os.path.splitext(files)[1]
                        # if ext not in valid_ext:
                        #       continue
                        # load image
                        input_image = Image.open(files).convert("RGB")
                        # resize image, keep aspect ratio
                        h = input_image.size[0]
                        w = input_image.size[1]
                        ratio = h *1.0 / w
                        if ratio > 1:
                                h = opt.load_size
                                w = int(h*1.0/ratio)
                        else:
                                w = opt.load_size
                                h = int(w * ratio)
                        input_image = input_image.resize((h, w), Image.BICUBIC)
                        input_image = np.asarray(input_image)
                        # RGB -> BGR
                        input_image = input_image[:, :, [2, 1, 0]]
                        input_image = transforms.ToTensor()(input_image).unsqueeze(0)
                        # preprocess, (-1, 1)
                        input_image = -1 + 2 * input_image
                        if opt.gpu > -1:
                                input_image = Variable(input_image, volatile=True).cuda()
                        else:
                                input_image = Variable(input_image, volatile=True).float()
                        # forward
                        output_image = model(input_image)
                        output_image = output_image[0]
                        # BGR -> RGB
                        output_image = output_image[[2, 1, 0], :, :]
                        # deprocess, (0, 1)
                        output_image = output_image.data.cpu().float() * 0.5 + 0.5
                        # save
                        vutils.save_image(output_image, os.path.join(savedir, files.split('/')[-1][:-4] + '_' + opt.style.lower() + '.jpg'))

if dataset == 'imagenet':
        fold = DATA + '/imagenet-lts/original'
        print(fold)

        filelist = glob.glob(fold + '/*JPEG')
        savedir = fold.replace('original', opt.style.lower())

        if not os.path.isdir(savedir):
                os.makedirs(savedir)

        for files in tqdm(filelist):
                input_image = Image.open(files).convert("RGB")
                # resize image, keep aspect ratio
                h = input_image.size[0]
                w = input_image.size[1]
                ratio = h * 1.0 / w
                if ratio > 1:
                        h = opt.load_size
                        w = int(h * 1.0 / ratio)
                else:
                        w = opt.load_size
                        h = int(w * ratio)
                input_image = input_image.resize((h, w), Image.BICUBIC)
                input_image = np.asarray(input_image)
                # RGB -> BGR
                input_image = input_image[:, :, [2, 1, 0]]
                input_image = transforms.ToTensor()(input_image).unsqueeze(0)
                # preprocess, (-1, 1)
                input_image = -1 + 2 * input_image
                if opt.gpu > -1:
                        input_image = Variable(input_image, volatile=True).cuda()
                else:
                        input_image = Variable(input_image, volatile=True).float()
                # forward
                output_image = model(input_image)
                output_image = output_image[0]
                # BGR -> RGB
                output_image = output_image[[2, 1, 0], :, :]
                # deprocess, (0, 1)
                output_image = output_image.data.cpu().float() * 0.5 + 0.5
                # save
                vutils.save_image(output_image, os.path.join(savedir, files.split('/')[-1][
                                                                      :-5] + '_' + opt.style.lower() + '.jpg'))

print('Done!')
