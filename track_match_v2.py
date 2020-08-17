# a combination of track and match
# 1. load fullres images, resize to 640**2
# 2. warmup: set random location for crop
# 3. loc-match: add attention
import os
import cv2
import sys
import time
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
from libs.loader import VidListv3
import torch.backends.cudnn as cudnn
from model import CRWNet as Model
from libs.loss import L1_loss
from libs.train_utils import save_vis, AverageMeter, save_checkpoint, log_current
from libs.utils import diff_crop


FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

############################## helper functions ##############################

def parse_args():
    parser = argparse.ArgumentParser(description='')

    # file/folder pathes
    parser.add_argument("--videoRoot", type=str, default="/data/home/v-yansta/train_256/", help='train video path')
    parser.add_argument("--videoList", type=str, default="/data/home/v-yansta/train_256.txt", help='train video list (after "train_256")')
    parser.add_argument('--resume', type=str, default='', metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument("-c","--savedir",type=str,default="results/crw",help='checkpoints path')
    parser.add_argument("--Resnet", type=str, default="r18", help="choose from r18 or r50")

    # main parameters
    parser.add_argument("--pretrainRes",action="store_true")
    parser.add_argument("--batchsize",type=int, default=24, help="batchsize")
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=64, help="crop size for localization.")
    parser.add_argument("--full_size", type=int, default=256, help="full size for one frame.")
    parser.add_argument("--lr",type=float,default=0.0001,help='learning rate')
    parser.add_argument('--lr-mode', type=str, default='poly')
    parser.add_argument("--video_len",type=int,default=10,help='length of input video')
    parser.add_argument("--fps",type=int,default=8,help='length of input video')
    parser.add_argument("--log_interval",type=int,default=10,help='')
    parser.add_argument("--save_interval",type=int,default=1000,help='save every x epoch')
    parser.add_argument("--momentum",type=float,default=0.9,help='momentum')
    parser.add_argument("--weight_decay",type=float,default=0.005,help='weight decay')
    parser.add_argument("--device", type=int, nargs="+", default=[0], help="device numbers. Multiple numbers for dataparallel")
    parser.add_argument("--temp", type=float, default=1.0, help="temprature for softmax.")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate for transition matrix")
    parser.add_argument("--feature_dim", type=int, default=128, help="Dimension of feature")

    # set epoches
    parser.add_argument("--nepoch",type=int,default=20,help='max epoch')

    
    print("Begin parser arguments.")
    args = parser.parse_args()
    assert args.videoRoot is not None
    assert args.videoList is not None
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    args.savepatch = os.path.join(args.savedir,'savepatch')
    args.logfile = open(os.path.join(args.savedir,"logargs.txt"),"w") 
    args.multiGPU = len(args.device) > 1

    if not args.multiGPU:
        torch.cuda.set_device(args.device[0])
    if not os.path.exists(args.savepatch):
        os.mkdir(args.savepatch)

    args.vis = True
    try:
        from tensorboardX import SummaryWriter
        global writer
        writer = SummaryWriter(logdir=os.path.join(args.savedir, 'tb_logger'))
        logger.info('Use tensorboardX')
    except ImportError:
        args.vis = False
    print(' '.join(sys.argv))
    print('\n')
    args.logfile.write(' '.join(sys.argv))
    args.logfile.write('\n')
    
    for k, v in args.__dict__.items():
        print(k, ':', v)
        args.logfile.write('{}:{}\n'.format(k,v))
    args.logfile.close()
    return args
    

def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.nepoch) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
    

def create_loader(args):
    dataset_train = VidListv3(args.videoRoot, args.videoList, args.full_size, args.patch_size, args.video_len, args.fps)

    if args.multiGPU:
        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batchsize, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batchsize, shuffle=True, num_workers=args.workers, drop_last=True)
    return train_loader


def train(args):
    loader = create_loader(args)
    cudnn.benchmark = True
    best_loss = 1e10
    start_epoch = 0

    model = Model(args.feature_dim, args.pretrainRes, args.Resnet, args.dropout, args.temp)

    cri = nn.NLLLoss()
    if args.multiGPU:
        model = torch.nn.DataParallel(model, device_ids=args.device).cuda()
    else:
        model.cuda()
    cri.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),args.lr)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{} ({})' (epoch {})"
                  .format(args.resume, best_loss, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))     
            
    for epoch in range(start_epoch, args.nepoch):
        lr = adjust_learning_rate(args, optimizer, epoch)
        print("Base lr for epoch {}: {}.".format(epoch, optimizer.param_groups[0]['lr']))
        best_loss = train_iter(args, loader, model, cri, optimizer, epoch, best_loss)

def train_iter(args, loader, model, cri, optimizer, epoch, best_loss):
    losses = AverageMeter()
    batch_time = AverageMeter()
    model.train()
    end = time.time()
    flag = True
        
    for i, (_, patch_sample) in enumerate(loader):
        # B*T*P*C*h*w
        patch_sample = patch_sample.cuda()
        if flag:
            label = torch.arange(start=0, end=patch_sample.size(2))
            # B*P
            label = label.long().unsqueeze(0).repeat(patch_sample.size(0), 1).cuda()
            flag = False
        
        _, At = model(patch_sample)
        loss = []
        for A in At:
            loss.append(cri(A, label))

        all_loss = sum(loss) / len(loss)
        losses.update(all_loss.item())
        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        #import pdb; pdb.set_trace()
        message = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i+1, len(loader))
        message +='Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(batch_time=batch_time)
        message +='Cycle Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss=losses)
        logger.info(message)

        if((i + 1) % args.save_interval == 0):
            is_best = losses.avg < best_loss
            best_loss = min(losses.avg, best_loss)
            checkpoint_path = os.path.join(args.savedir, 'checkpoint_latest.pth.tar')
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                }, is_best, filename=checkpoint_path, savedir = args.savedir)
            log_current(epoch, losses.avg, best_loss, filename = "log_current.txt", savedir=args.savedir)

    return best_loss


if __name__ == '__main__':
    args = parse_args()
    torch.autograd.set_detect_anomaly(True)
    train(args)
    writer.close()
