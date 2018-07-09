#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Sample code for inference of Progressive Growing of GANs paper
(https://github.com/tkarras/progressive_growing_of_gans)
using a CelebA snapshot
"""

from __future__ import print_function
import argparse

import torch
from torch.autograd import Variable

from model import Generator

from utils import scale_image

import os
from PIL import Image
import time


parser = argparse.ArgumentParser(description='Inference demo')
parser.add_argument(
    '--weights',
    default='100_celeb_hq_network-snapshot-010403.pth',
    type=str,
    metavar='PATH',
    help='path to PyTorch state dict')
parser.add_argument('--cuda', dest='cuda', action='store_true')

def run(model, path):
    global use_cuda
    
    for i in xrange(0, 50):
        # Generate latent vector
        x = torch.randn(1, 512, 1, 1)
        if use_cuda:
            model = model.cuda()
            x = x.cuda()
        x = Variable(x, volatile=True)
        image = model(x)
        if use_cuda:
            image = image.cpu()
        
        image_np = image.data.numpy().transpose(0, 2, 3, 1)
        image_np = scale_image(imags_np[0, ...])
        image = Image.fromarray(image_np)
        fname = os.path.join(path, '_gen{}.jpg'.format(i))
        image.save(fname)
        print("{}th image".format(i))



def main():
    global use_cuda
    args = parser.parse_args()

    if not args.weights:
        print('No PyTorch state dict path privided. Exiting...')
        return
    
    if args.cuda:
        use_cuda = True
        print("Use cuda")

    seed = int(time.time())
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)

    print('Loading Generator')
    model = Generator()
    model.load_state_dict(torch.load(args.weights))

    # create folder.
    for i in range(1000):
        name = 'repo/generate/try_{}'.format(i)
        if not os.path.exists(name):
            os.system('mkdir -p {}'.format(name))
            break

    run(model, name)


if __name__ == '__main__':
    main()