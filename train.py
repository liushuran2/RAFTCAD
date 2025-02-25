from __future__ import print_function, division
import sys

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,7'
# 
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import shutil

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import Subset, SubsetRandomSampler, DataLoader


# implemented
from model.raft import RAFT
from dataset.datasets import *
from engine import *
from model.loss import sequence_loss, sequence_maskedloss


# logging related
import socket
from datetime import datetime, timedelta

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler

except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def train(args):
    # debug purpose
    # args.wandb_flag = False
    # args.epochs = 5

    # fix random seed
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    ################## setup dataloader  
    # train dataset  
    aug_params = {'crop_size': args.aug_image_size, 'min_scale': args.aug_min_scale, 'max_scale': args.aug_max_scale, 'do_flip': args.aug_flip, 'do_rotate': args.aug_rotate}
    # aug_params = None
    train_dataset = FlowDataset(args, args.data_path, aug_params)

    # Split train and validation datasets
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(args.val_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    # Create train and validation samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create train and validation dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, sampler=train_sampler, shuffle=False,
                                                   num_workers=args.workers, pin_memory=True, persistent_workers=True,
                                                   drop_last=True, worker_init_fn=worker_init_fn)


    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, sampler=train_sampler,
    #                                                worker_init_fn=worker_init_fn)
    val_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, sampler=val_sampler,
                                                 num_workers=1, pin_memory=True, persistent_workers=True,
                                                 drop_last=True, worker_init_fn=worker_init_fn)

    # test dataset, no shuffle. 
    test_dataloader_array = []
    args.data_property  = []
    for datapath in  args.test_file:
        test_dataset = FlowTestDataset(args, datapath)
        args.data_property.append(test_dataset.data_property) # call the data_property for getting the data info
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, # note shuffle is disabled, and batchsize has to be 1
                                                        num_workers=args.workers, pin_memory=True, persistent_workers=True,
                                                        drop_last=False, worker_init_fn=worker_init_fn)
        test_dataloader_array.append(test_dataloader)
    ################## Building model
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    # model = RAFT(args)
    # model = SPyNet('https://download.openmmlab.com/mmediting/restorers/''basicvsr/spynet_20210409-c6c1bd09.pth')
    total_param =  count_parameters(model)
    print("Parameter Count: %d" % count_parameters(model))

    # model parameters
    param_str = "Parameter Count: %d" % total_param

    print(f'{args}\n {model}\n {param_str}', flush=True)
    with open('{}/log.txt'.format(args.outf), 'a') as f:
        f.write(str(model) + '\n' + f'{param_str}\n')

    # to GPU
    model.cuda()

    # set to train mode
    model.train()

    # optimizer and scheduler
    args.num_steps = len(train_dataloader) * args.epochs
    print('total steps: ', args.num_steps)
    optimizer, scheduler = fetch_optimizer(args, model) # scheduler use step information

    # scalar
    scaler = GradScaler(enabled=args.mixed_precision) # mixed precision training

    # logger
    writer = SummaryWriter(os.path.join(args.outf, 'tensorboard'))

    # criterion
    criterion = sequence_loss

    ################## resume from model_latest
    checkpoint = None
    if not args.not_resume:
        checkpoint_path = os.path.join(args.outf, 'model_latest.pth')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            print("=> Auto resume loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> No resume checkpoint found at '{}'".format(checkpoint_path))

    if args.start_epoch < 0:
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] 
        args.start_epoch = max(args.start_epoch, 0)

    ################## Training
    start = datetime.now()
    best_pred_epe = float('inf')
    time_per_epoch_list = []
    for epoch in range(args.start_epoch, args.epochs): # loop over the dataset multiple times
        epoch_start_time = time.time()

        # train one epoch
        train_loss_list, train_epe_list = train_one_epoch(model, optimizer, criterion, scheduler, train_dataloader, scaler, epoch, args, start, writer)
        average_epe = np.mean(np.array(train_epe_list))
        average_loss = np.mean(np.array(train_loss_list))
            
        # save model. 
        state_dict = model.state_dict()
        save_checkpoint = {
            'epoch': epoch+1,
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict(),   
        }    

        # evaluation, since this is time-consuming, we dont evaluate every epoch
        if (epoch + 1) % args.eval_freq == 0 or (args.epochs - epoch) in [1, 3, 5]:
            eval_loss_list, eval_epe_list, eval_data_list = evaluate(model, criterion, val_dataloader, args, epoch, writer)

            average_eval_epe = np.mean(np.array(eval_epe_list))
            average_eval_loss = np.mean(np.array(eval_loss_list))
            average_data_loss = np.mean(np.array(eval_data_list))
            # buiild print string
            writer.add_scalar(f'Val/epe', average_eval_epe, epoch+1)
            writer.add_scalar(f'Val/loss', average_eval_loss, epoch+1)
            writer.add_scalar(f'Val/data_loss', average_data_loss, epoch+1)

            # save best based on metrics
            if average_eval_epe < best_pred_epe:
                best_pred_epe = average_eval_epe
                torch.save(save_checkpoint, f'{args.outf}/model_best.pth') 
        # save the last
        torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.outf))
        if (epoch + 1) % args.epochs == 0:
            args.cur_epoch = epoch + 1
            args.train_time = str(datetime.now() - start)
            torch.save(save_checkpoint, f'{args.outf}/epoch{epoch+1}.pth')

        # time estimation
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        time_per_epoch_list.append(epoch_duration)
        
        average_time_per_epoch = sum(time_per_epoch_list) / len(time_per_epoch_list)
        estimated_remaining_time = average_time_per_epoch * (args.epochs - epoch - 1)
        
        finish_time = datetime.now() + timedelta(seconds=estimated_remaining_time)
        print(f"Estimated Finish Time: {finish_time.strftime('%Y-%m-%d %H:%M:%S')}")

        
    ############# do the testing after the training
    for i, (test_dataloader, test_file_name) in enumerate(zip(test_dataloader_array, args.test_file)):
        session_name = test_file_name.split('/')[-1].split('.')[0]
        test(model, args, test_dataloader, session_name, args.data_property[i], iters=32, warm_start=False, output_path=args.outf)
    
    writer.close()  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # high-level settings
    parser.add_argument('--project_name', type=str, default='RAFT', help='this used for wandb saving')
    parser.add_argument('--data_path', type=str, default='/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200/N_3600_scale_10_stack_8_multiscale_10mW.h5', help='data path for vid')
    parser.add_argument('--outf', type=str, default='/mnt/nas01/LSR/DATA/checkpt/RAFTCAD_result_multiscale_stack_3600', help='data path for vid')
    # parser.add_argument('--data_path', type=str, default='../../Dataset/Gen_motion_free_pair_frame/N_99_scale_10.h5', help='data path for vid')
    parser.add_argument('--norm_type', type=str, default='robust', help='video normalization methods')
    parser.add_argument('--wandb_flag', type=bool, default=False)
    parser.add_argument('-p', '--print-freq', default=100, type=int)
    # test file
    parser.add_argument('--test_file', type=str, nargs='+', default=[], 
                        help='test file for evaluation')
    # parser.add_argument('--test_file', type=str, nargs='+', default=['../../Dataset/exp_dataset/148d_1.tif', 
    #                                                                  '../../Dataset/exp_dataset/148d_2.tif', ], 
    #                     help='test file for evaluation')

    # dataset and data augmentation
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('-j', '--workers', type=int, default=1, help='number of data loading workers')
    parser.add_argument('--aug_image_size', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--aug_min_scale', type=float, default=0.2, help='Minimum scale for augmentation')
    parser.add_argument('--aug_max_scale', type=float, default=0.5, help='Maximum scale for augmentation')
    parser.add_argument('--aug_flip', type=bool, default=True, help='Whether to flip the image for augmentation')
    parser.add_argument('--aug_rotate', type=bool, default=True, help='Whether to rotate the image for augmentation')

    # General training setups
    parser.add_argument('--manualSeed', type=int, default=13, help='manual seed')
    parser.add_argument('--start_epoch', type=int, default=-1, help='starting epoch')
    parser.add_argument('-e', '--epochs', type=int, default=600, help='Epoch number')
    parser.add_argument('--eval_freq', type=int, default=5, help='evaluation frequency,  added to suffix!!!!')
    parser.add_argument('--not_resume', action='store_true', default = False, help='not resume from latest checkpoint')

    parser.add_argument('-b', '--batchSize', type=int, default=3, help='input batch size')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1,2])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # optimizer, for AdamW
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)

    # gradient clipping
    parser.add_argument('--clip', type=float, default=1.0)

    # dropout
    parser.add_argument('--dropout', type=float, default=0.0)

    # loss
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')

    # noise
    parser.add_argument('--add_noise', default=False, action='store_true')
    parser.add_argument('--add_blur', default=False, action='store_true')

    # model config
    parser.add_argument('--small', action='store_true', default=False, help='use small model')
    parser.add_argument('--iters', type=int, default=12)

    # parser
    args = parser.parse_args()


    # platform determined saving
    # hostname = socket.gethostname()
    # if hostname == "YZ-win-station":
    #     args.outf = 'U:/Projects/Project_DeepMotionReg/results/'
    # elif hostname == "YZ-Ghost-S1":
    #     args.outf = 'E:/Project/Project_DeepMotionReg/'
    # elif hostname == 'bbnc-2':
    #     args.outf = 'out/'
    # elif hostname == 'bbnc-System-Product-Name':
    #     args.outf = 'out/'
    # elif hostname == 'bbnc-1':
    #     args.outf = 'out/'
    # else:
    #     NameError("Unknown computer")

    # saving path
    timestamp = datetime.now().strftime("%Y-%m-%d")
    parts = args.data_path.split('/')
    d_path = parts[-1] if len(parts) > 1 else ''
    
 
    # args.outf = args.outf + f'{timestamp}' # add model size and date to the output folder

    # make the directory
    # if os.path.isdir(args.outf):
    #     print('Will overwrite the existing output dir!')
    #     shutil.rmtree(args.outf)

    if not os.path.isdir(args.outf):
        os.makedirs(args.outf) 
    # save args for next loading
    save_args(args, filename=f'{args.outf}/args.json')
    # time.sleep(15 * 60)

    # do the training
    train(args)