#!/user/bin/python
# coding=utf-8
import os, sys
import numpy as np
from PIL import Image
import cv2
import shutil
import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import BSDS_RCFLoader
from models import RCF
from functions import  cross_entropy_loss_RCF, SGD_caffe
from torch.utils.data import DataLoader, sampler
from utils import Logger, Averagvalue, save_checkpoint, load_vgg16pretrain
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=1, type=int, metavar='BT',
                    help='batch size')
# =============== optimizer
parser.add_argument('--lr', '--learning_rate', default=1e-6, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=3, type=int, 
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--maxepoch', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--itersize', default=10, type=int,
                    metavar='IS', help='iter size')
# =============== misc
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=1000, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU ID')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--tmp', help='tmp folder', default='tmp/RCF')
# ================ dataset
parser.add_argument('--dataset', help='root folder of dataset', default='data/HED-BSDS_PASCAL')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

THIS_DIR = abspath(dirname(__file__))
TMP_DIR = join(THIS_DIR, args.tmp)
if not isdir(TMP_DIR):
  os.makedirs(TMP_DIR)
print('***', args.lr)
def main():
    args.cuda = True
    # dataset
    train_dataset = BSDS_RCFLoader(root=args.dataset, split="train")
    test_dataset = BSDS_RCFLoader(root=args.dataset, split="test")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=8, drop_last=True,shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers=8, drop_last=True,shuffle=False)
    with open('data/HED-BSDS_PASCAL/test.lst', 'r') as f:
        test_list = f.readlines()
    test_list = [split(i.rstrip())[1] for i in test_list]
    assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))

    # model
    model = RCF()
    model.cuda()
    model.apply(weights_init)
    load_vgg16pretrain(model)
    if args.resume:
        if isfile(args.resume): 
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    #tune lr
    net_parameters_id = {}
    net = model
    for pname, p in net.named_parameters():
        if pname in ['conv1_1.weight','conv1_2.weight',
                     'conv2_1.weight','conv2_2.weight',
                     'conv3_1.weight','conv3_2.weight','conv3_3.weight',
                     'conv4_1.weight','conv4_2.weight','conv4_3.weight']:
            print(pname, 'lr:1 de:1')
            if 'conv1-4.weight' not in net_parameters_id:
                net_parameters_id['conv1-4.weight'] = []
            net_parameters_id['conv1-4.weight'].append(p)
        elif pname in ['conv1_1.bias','conv1_2.bias',
                       'conv2_1.bias','conv2_2.bias',
                       'conv3_1.bias','conv3_2.bias','conv3_3.bias',
                       'conv4_1.bias','conv4_2.bias','conv4_3.bias']:
            print(pname, 'lr:2 de:0')
            if 'conv1-4.bias' not in net_parameters_id:
                net_parameters_id['conv1-4.bias'] = []
            net_parameters_id['conv1-4.bias'].append(p)
        elif pname in ['conv5_1.weight','conv5_2.weight','conv5_3.weight']:
            print(pname, 'lr:100 de:1')
            if 'conv5.weight' not in net_parameters_id:
                net_parameters_id['conv5.weight'] = []
            net_parameters_id['conv5.weight'].append(p)
        elif pname in ['conv5_1.bias','conv5_2.bias','conv5_3.bias'] :
            print(pname, 'lr:200 de:0')
            if 'conv5.bias' not in net_parameters_id:
                net_parameters_id['conv5.bias'] = []
            net_parameters_id['conv5.bias'].append(p)
        elif pname in ['conv1_1_down.weight','conv1_2_down.weight',
                       'conv2_1_down.weight','conv2_2_down.weight',
                       'conv3_1_down.weight','conv3_2_down.weight','conv3_3_down.weight',
                       'conv4_1_down.weight','conv4_2_down.weight','conv4_3_down.weight',
                       'conv5_1_down.weight','conv5_2_down.weight','conv5_3_down.weight']:
            print(pname, 'lr:0.1 de:1')
            if 'conv_down_1-5.weight' not in net_parameters_id:
                net_parameters_id['conv_down_1-5.weight'] = []
            net_parameters_id['conv_down_1-5.weight'].append(p)
        elif pname in ['conv1_1_down.bias','conv1_2_down.bias',
                       'conv2_1_down.bias','conv2_2_down.bias',
                       'conv3_1_down.bias','conv3_2_down.bias','conv3_3_down.bias',
                       'conv4_1_down.bias','conv4_2_down.bias','conv4_3_down.bias',
                       'conv5_1_down.bias','conv5_2_down.bias','conv5_3_down.bias']:
            print(pname, 'lr:0.2 de:0')
            if 'conv_down_1-5.bias' not in net_parameters_id:
                net_parameters_id['conv_down_1-5.bias'] = []
            net_parameters_id['conv_down_1-5.bias'].append(p)
        elif pname in ['score_dsn1.weight','score_dsn2.weight','score_dsn3.weight',
                       'score_dsn4.weight','score_dsn5.weight']:
            print(pname, 'lr:0.01 de:1')
            if 'score_dsn_1-5.weight' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5.weight'] = []
            net_parameters_id['score_dsn_1-5.weight'].append(p)
        elif pname in ['score_dsn1.bias','score_dsn2.bias','score_dsn3.bias',
                       'score_dsn4.bias','score_dsn5.bias']:
            print(pname, 'lr:0.02 de:0')
            if 'score_dsn_1-5.bias' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5.bias'] = []
            net_parameters_id['score_dsn_1-5.bias'].append(p)
        elif pname in ['score_final.weight']:
            print(pname, 'lr:0.001 de:1')
            if 'score_final.weight' not in net_parameters_id:
                net_parameters_id['score_final.weight'] = []
            net_parameters_id['score_final.weight'].append(p)
        elif pname in ['score_final.bias']:
            print(pname, 'lr:0.002 de:0')
            if 'score_final.bias' not in net_parameters_id:
                net_parameters_id['score_final.bias'] = []
            net_parameters_id['score_final.bias'].append(p)

    optimizer = torch.optim.SGD([
            {'params': net_parameters_id['conv1-4.weight']      , 'lr': args.lr*1    , 'weight_decay': args.weight_decay},
            {'params': net_parameters_id['conv1-4.bias']        , 'lr': args.lr*2    , 'weight_decay': 0.},
            {'params': net_parameters_id['conv5.weight']        , 'lr': args.lr*100  , 'weight_decay': args.weight_decay},
            {'params': net_parameters_id['conv5.bias']          , 'lr': args.lr*200  , 'weight_decay': 0.},
            {'params': net_parameters_id['conv_down_1-5.weight'], 'lr': args.lr*0.1  , 'weight_decay': args.weight_decay},
            {'params': net_parameters_id['conv_down_1-5.bias']  , 'lr': args.lr*0.2  , 'weight_decay': 0.},
            {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': args.lr*0.01 , 'weight_decay': args.weight_decay},
            {'params': net_parameters_id['score_dsn_1-5.bias']  , 'lr': args.lr*0.02 , 'weight_decay': 0.},
            {'params': net_parameters_id['score_final.weight']  , 'lr': args.lr*0.001, 'weight_decay': args.weight_decay},
            {'params': net_parameters_id['score_final.bias']    , 'lr': args.lr*0.002, 'weight_decay': 0.},
        ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)


    # optimizer = torch.optim.Adam([
    #         {'params': net_parameters_id['conv1-4.weight']      , 'lr': args.lr*1    , 'weight_decay': args.weight_decay},
    #         {'params': net_parameters_id['conv1-4.bias']        , 'lr': args.lr*2    , 'weight_decay': 0.},
    #         {'params': net_parameters_id['conv5.weight']        , 'lr': args.lr*100  , 'weight_decay': args.weight_decay},
    #         {'params': net_parameters_id['conv5.bias']          , 'lr': args.lr*200  , 'weight_decay': 0.},
    #         {'params': net_parameters_id['conv_down_1-5.weight'], 'lr': args.lr*0.1  , 'weight_decay': args.weight_decay},
    #         {'params': net_parameters_id['conv_down_1-5.bias']  , 'lr': args.lr*0.2  , 'weight_decay': 0.},
    #         {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': args.lr*0.01 , 'weight_decay': args.weight_decay},
    #         {'params': net_parameters_id['score_dsn_1-5.bias']  , 'lr': args.lr*0.02 , 'weight_decay': 0.},
    #         {'params': net_parameters_id['score_final.weight']  , 'lr': args.lr*0.001, 'weight_decay': args.weight_decay},
    #         {'params': net_parameters_id['score_final.bias']    , 'lr': args.lr*0.002, 'weight_decay': 0.},
    #     ], lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    
    # log
    log = Logger(join(TMP_DIR, '%s-%d-log.txt' %('sgd',args.lr)))
    sys.stdout = log

    train_loss = []
    train_loss_detail = []
    for epoch in range(args.start_epoch, args.maxepoch):
        if epoch == 0:
            print("Performing initial testing...")
            multiscale_test(model, test_loader, epoch=epoch, test_list=test_list,
                 save_dir = join(TMP_DIR, 'initial-testing-record'))

        tr_avg_loss, tr_detail_loss = train(
            train_loader, model, optimizer, epoch,
            save_dir = join(TMP_DIR, 'epoch-%d-training-record' % epoch))
        test(model, test_loader, epoch=epoch, test_list=test_list,
            save_dir = join(TMP_DIR, 'epoch-%d-testing-record-view' % epoch))
        multiscale_test(model, test_loader, epoch=epoch, test_list=test_list,
            save_dir = join(TMP_DIR, 'epoch-%d-testing-record' % epoch))
        log.flush() # write log
        # Save checkpoint
        save_file = os.path.join(TMP_DIR, 'checkpoint_epoch{}.pth'.format(epoch))
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
                         }, filename=save_file)
        scheduler.step() # will adjust learning rate
        # save train/val loss/accuracy, save every epoch in case of early stop
        train_loss.append(tr_avg_loss)
        train_loss_detail += tr_detail_loss

def train(train_loader, model, optimizer, epoch, save_dir):
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    # switch to train mode
    model.train()
    end = time.time()
    epoch_loss = []
    counter = 0
    for i, (image, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        image, label = image.cuda(), label.cuda()
        outputs = model(image)
        loss = torch.zeros(1).cuda()
        for o in outputs:
            loss = loss + cross_entropy_loss_RCF(o, label)
        counter += 1
        loss = loss / args.itersize
        loss.backward()
        if counter == args.itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))
        epoch_loss.append(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        # display and logging
        if not isdir(save_dir):
            os.makedirs(save_dir)
        if i % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, i, len(train_loader)) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(
                       loss=losses)
            print(info)
            label_out = torch.eq(label, 1).float()
            outputs.append(label_out)
            _, _, H, W = outputs[0].shape
            all_results = torch.zeros((len(outputs), 1, H, W))
            for j in range(len(outputs)):
                all_results[j, 0, :, :] = outputs[j][0, 0, :, :]
            torchvision.utils.save_image(1-all_results, join(save_dir, "iter-%d.jpg" % i))
        # save checkpoint
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
            }, filename=join(save_dir, "epoch-%d-checkpoint.pth" % epoch))

    return losses.avg, epoch_loss

def test(model, test_loader, epoch, test_list, save_dir):
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    for idx, image in enumerate(test_loader):
        image = image.cuda()
        _, _, H, W = image.shape
        results = model(image)
        result = torch.squeeze(results[-1].detach()).cpu().numpy()
        results_all = torch.zeros((len(results), 1, H, W))
        for i in range(len(results)):
          results_all[i, 0, :, :] = results[i]
        filename = splitext(test_list[idx])[0]
        torchvision.utils.save_image(1-results_all, join(save_dir, "%s.jpg" % filename))
        result = Image.fromarray((result * 255).astype(np.uint8))
        result.save(join(save_dir, "%s.png" % filename))
        print("Running test [%d/%d]" % (idx + 1, len(test_loader)))
# torch.nn.functional.upsample(input, size=None, scale_factor=None, mode='nearest', align_corners=None)
def multiscale_test(model, test_loader, epoch, test_list, save_dir):
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    scale = [0.5, 1, 1.5]
    for idx, image in enumerate(test_loader):
        image = image[0]
        image_in = image.numpy().transpose((1,2,0))
        _, H, W = image.shape
        multi_fuse = np.zeros((H, W), np.float32)
        for k in range(0, len(scale)):
            im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2,0,1))
            results = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
            result = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
            multi_fuse += fuse
        multi_fuse = multi_fuse / len(scale)
        ### rescale trick suggested by jiangjiang
        # multi_fuse = (multi_fuse - multi_fuse.min()) / (multi_fuse.max() - multi_fuse.min())
        filename = splitext(test_list[idx])[0]
        result_out = Image.fromarray(((1-multi_fuse) * 255).astype(np.uint8))
        result_out.save(join(save_dir, "%s.jpg" % filename))
        result_out_test = Image.fromarray((multi_fuse * 255).astype(np.uint8))
        result_out_test.save(join(save_dir, "%s.png" % filename))
        print("Running test [%d/%d]" % (idx + 1, len(test_loader)))

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()

if __name__ == '__main__':
    main()
