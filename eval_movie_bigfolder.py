
from __future__ import print_function, division
import sys

import argparse
import configparser
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import shutil
import yaml

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import Subset, SubsetRandomSampler, DataLoader
import scipy
from scipy import io
# implemented
from model.raft import RAFT
from dataset.datasets import *
from engine import *
from model.loss import sequence_loss
from tensorRT.loadengine import load_engine


# logging related
# import wandb 
# from wandb import sdk as wanbd_sdk
import socket
from datetime import datetime, timedelta

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = '6'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # test file
    parser.add_argument('--model_path', default='/home/shuran/RAFTCADSUN/checkpt/RAFTCAD_result_multiscale_stack_2002/', help='path to the trained model')
    parser.add_argument('--gt_flow', type=str, nargs='+', default=None, 
                        help='test file for evaluation')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--add_blur', default=False, action='store_true')
    parser.add_argument('--doublestage', default=True, action='store_true')

    # parser
    args_eval = parser.parse_args()

    # load the flow
    if args_eval.gt_flow is not None:
        args_eval.gt_flow = scipy.io.loadmat(args_eval.gt_flow[0])['motion_field']
    else:
        args_eval.gt_flow = None

    print(os.path.join(args_eval.model_path, 'args.json'))
    # Create a ConfigParser object
    tmp = FlexibleNamespace()
    # if os.path.exists(os.path.join(args_eval.model_path, 'args.json')):
    args_model = tmp.load_from_json(os.path.join(args_eval.model_path, 'args.json'))
    print_args_formatted(args_model)
    # else:
    #     print("No args.json file found.")
    #     sys.exit()
    
    # get the output path
    outf = args_eval.model_path


    # load the network
    checkpoint_path = os.path.join(outf, 'model_latest.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = nn.DataParallel(RAFT(args_model))
        # model = SPyNet('https://download.openmmlab.com/mmediting/restorers/''basicvsr/spynet_20210409-c6c1bd09.pth')
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        model.eval()

    trt_path = '/mnt/nas01/LSR/DATA/checkpt/RAFTCAD_result_multiscale_stack_3600_50mW/DeepIE_tensorRT.trt'
    engine = load_engine(trt_path)
    if engine is None:
        print("加载引擎失败。")

    # get the evaluating dataset
    test_dataloader_array = []
    args_eval.data_property  = []
    args_eval.norm_type = args_model.norm_type

    for directory_id in range(1):
        # directory_path = '/mnt/nas/YZ_personal_storage/Private/MC/NAOMi_2p_4/test_seq/' + str(directory_id+1) + '/'
        # loadframe_path = directory_path
        # directory_path = '/mnt/nas/YZ_personal_storage/Private/MC/mini2p/ca1/' + str(directory_id+1) + '/'
        # directory_path = '/mnt/nas/YZ_personal_storage/Private/MC/2p_fiberscopy/' + str(directory_id+1)+ '/'
        # directory_path = '/mnt/nas00/YZ_personal_storage/Private/MC/2p_benchtop/2p_148d/trial_2p_' + str(directory_id+1) + '/'
        # directory_path = 'DATA/'
        directory_path = '/mnt/nas01/LSR/DATA/2p_bench/HP01/HCA301-Axon-P5-oir/HCA301-Axon-P5-oir/'
        # directory_path = '/mnt/nas00/YZ_personal_storage/Private/MC/simulation_noise/90W_10scale_5range/'
        # directory_path = '/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200_test/test_dataset/10mW/'
        loadframe_path = directory_path
        # loadframe_path = directory_path + 'Self_Rigid_result/'
        all_files = os.listdir(loadframe_path)
        
    
        tiff_files = [filename for filename in all_files if filename.endswith('-1.tif')]
        tiff_files.sort()

        save_path = directory_path + 'DeepIE_50mW/'
        # save_path = '/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200_test/result/DeepIE/90mWwith10mW/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for fnames in tiff_files:
            session_name = fnames.split('/')[-1].split('.')[0]
            datapath = loadframe_path + fnames
            test_dataset = FlowTestDataset(args_model, datapath)
            args_eval.data_property = test_dataset.data_property
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, # note shuffle is disabled, and batchsize has to be 1
                                                        num_workers=args_model.workers, pin_memory=True, persistent_workers=True,
                                                        drop_last=False, worker_init_fn=worker_init_fn)
            # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # starter.record()
            if args_eval.doublestage:
                iters = 1
            else:
                iters = 2
            # test(model, args_eval, test_dataloader, session_name, args_eval.data_property, 
            #      iters=iters, warm_start=False, output_path=save_path)

            test_tensorRT(engine, args_eval, test_dataloader, session_name, args_eval.data_property, output_path=save_path)
        print(directory_id)
            # ender.record()
            # curr_time = starter.elapsed_time(ender)
            # print(curr_time)


