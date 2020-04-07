import os
import time
import random
import argparse
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import video_transforms
import models
import dataset

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in dataset.__all__)

parser = argparse.ArgumentParser(description='PyTorch Action Recognition')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--settings', metavar='DIR', default='./dataset/settings',
                    help='path to datset setting files')
parser.add_argument('--dataset', '-d', default='ucf101',
                    choices=["ucf101", "hmdb51"],
                    help='dataset: ucf101 | hmdb51')
parser.add_argument('--arch', '-a', metavar='ARCH', default='flow_resnet152',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: flow_resnet152)')
parser.add_argument('-s', '--split', default=2, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=25 , type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--iter-size', default=5, type=int,
                    metavar='I', help='iter size as in Caffe to reduce memory usage (default: 5)')
parser.add_argument('--n_images', default=1, type=int,
                    metavar='N', help='number of visual rhythm images per video (default: 1)')
parser.add_argument('--new_width', default=360, type=int,
                    metavar='N', help='resize width (default: 320,360)')
parser.add_argument('--new_height', default=320, type=int,
                    metavar='N', help='resize height (default: 240,320)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[100, 200], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)') 
parser.add_argument('-pf','--print-freq', default=50,  type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('-sf','--save-freq', default=25, type=int,
                    metavar='N', help='save frequency (default: 25)')
parser.add_argument('--resume', default='./checkpoints', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec1 = 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():
    global args, best_prec1
    args = parser.parse_args()

    print("Network trained whith the split "+str(args.split)+".")

    # create model
    print("Building model ... ")
    exits_model, model = build_model(int(args.start_epoch))
    if not exits_model:
        return 
    else:
        print("Model %s is loaded. " % (args.arch))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # create file where we allocate the models by each args.save_freq epochs
    if not os.path.exists(args.resume):
        os.makedirs(args.resume)
    print("Saving everything to directory %s." % (args.resume))

    cudnn.benchmark = True

    # Data transforming
    is_color = False
    scale_ratios = [1.0, 0.875, 0.75]
    clip_mean = [0.5]*args.n_images
    clip_std = [0.226]*args.n_images

    new_size= 299 if args.arch.find("inception_v3")>0 else 224

    normalize = video_transforms.Normalize(mean=clip_mean,
                                           std=clip_std)
    train_transform = video_transforms.Compose([
            video_transforms.RandomHorizontalFlip(),
            video_transforms.ToTensor(),
            normalize,
        ])

    val_transform = video_transforms.Compose([
            video_transforms.ToTensor(),
            normalize,
        ])

    # data loading  
    train_setting_file = "train_rgb_split%d.txt" % (args.split)
    train_split_file = os.path.join(args.settings, args.dataset, train_setting_file)
    val_setting_file = "val_rgb_split%d.txt" % (args.split) 
    val_split_file = os.path.join(args.settings, args.dataset, val_setting_file)

    if not os.path.exists(train_split_file) or not os.path.exists(val_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (args.settings))

    train_dataset = dataset.__dict__['dataset'](root=args.data,
                                                    source=train_split_file,
                                                    phase="train",
                                                    is_color=is_color,
                                                    n_images=args.n_images,
                                                    new_width=args.new_width,
                                                    new_height=args.new_height,
                                                    video_transform=train_transform)
    val_dataset = dataset.__dict__['dataset'](root=args.data,
                                                  source=val_split_file,
                                                  phase="val",
                                                  is_color=is_color,
                                                  n_images=args.n_images,
                                                  new_width=args.new_width,
                                                  new_height=args.new_height,
                                                  video_transform=val_transform) 

    print('{} samples found, {} train samples and {} test samples.'.format(len(val_dataset)+len(train_dataset),
                                                                           len(train_dataset),
                                                                           len(val_dataset)))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return
 
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = 0.0
        if (epoch + 1) % args.save_freq == 0:
            prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint_rhythm_split_"+str(args.split)+".pth.tar")
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_name, args.resume)
    
def build_model(resume_epoch): #OK
    is_new = (resume_epoch==0)
    found = True
    num_classes = 51 if args.dataset =='hmdb51' else 101
    num_channels = 1
    model = models.__dict__[args.arch](pretrained=is_new, channels=num_channels, num_classes=num_classes)
    if not is_new:
        path = os.path.join(args.resume,'%03d_checkpoint_rhythm_split_%s.pth.tar'%(resume_epoch,args.split))
        print(path)
        if os.path.isfile(path):    
            print('loading checkpoint {0:03d} ...'.format(resume_epoch))    
            params = torch.load(path)
            model.load_state_dict(params['state_dict'])
            print('loaded checkpoint {0:03d}'.format(resume_epoch))
        else:
            print('ERROR: No checkpoint found')
            found = False
    model.cuda()
    return found, model

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    loss_mini_batch = 0.0
    acc_mini_batch = 0.0

    for i, (input, target) in enumerate(train_loader):
        s1,s2 = input.shape[-2], input.shape[-1]
        input = input.view(-1,1,s1,s2)
        target = target.reshape(-1)

        input = input.float().cuda(async=True)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        outputs= model(input_var)

        #TODO voltar
        loss = None
        # for nets that have multiple outputs such as inception 
        if isinstance(outputs, tuple):
            loss = sum((criterion(o,target_var) for o in outputs))
            outputs_data = outputs[0].data
        else:
            loss = criterion(outputs, target_var)
            outputs_data = outputs.data    

        # measure accuracy and record loss
        prec1, prec3 = accuracy(outputs_data, target, topk=(1, 3))
        acc_mini_batch += prec1.item()
        loss = loss / args.iter_size
        loss_mini_batch += loss.data.item()
        loss.backward()

        if (i+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()

            losses.update(loss_mini_batch, input.size(0))
            top1.update(acc_mini_batch/args.iter_size, input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            loss_mini_batch = 0
            acc_mini_batch = 0

            if (i+1) % args.print_freq == 0:

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       epoch, i+1, len(train_loader)+1, batch_time=batch_time, loss=losses, top1=top1))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        s1,s2 = input.shape[-2], input.shape[-1]
        input = input.view(-1,1,s1,s2)
        target = target.reshape(-1)

        with torch.no_grad():
            input = input.float().cuda(async=True)
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top3.update(prec3.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top3=top3))

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))

    return top1.avg

def save_checkpoint(state, is_best, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    best_path = os.path.join(resume_path, 'model_best_rhythm_split_'+str(args.split)+'.pth.tar')
    torch.save(state, cur_path)
    if is_best:
        shutil.copyfile(cur_path, best_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch): #OK
    """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""

    decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
    lr = args.lr * decay
    print("Current learning rate is %4.6f:" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()

