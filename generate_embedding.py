import os
import tqdm
import argparse
import torch 
import torch.nn as nn
import numpy as np 


def generate():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--feature_file', help='the file save generated features')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = torch.load(args.load)
    