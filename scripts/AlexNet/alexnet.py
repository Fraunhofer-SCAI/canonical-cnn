# ==================================================================
# AlexNet trianing/Validaion procedure
# ==================================================================
import argparse
from collections import OrderedDict
import os
from os import cpu_count
from pathlib import Path
import random
import shutil
import sys
import time
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

import numpy as np
import tensorly as tl
tl.set_backend("pytorch")
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.approxlowrankmodel import tai_decompose
from src.cp_compress import apply_compression
from src.cp_norm import cp_norm
from models.AlexNet_model import AlexNet


parser = argparse.ArgumentParser(description='PyTorch Cifar-AlexNet Training')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.0, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0., type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training from a stored model.')
parser.add_argument('--name', default='exp1', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true',
                    default=True)
parser.add_argument('--cpu', help='Run on cpu only', action='store_true',
                    default=False)
parser.add_argument('--seed', type=int, default=int(time.time()), metavar='S',
                    help='random seed (default: time.time)')
parser.add_argument('--data_set', default='cifar10', type=str,
                    help='The data set to be used.')
parser.add_argument('--model', default='AlexNet', type=str,
                    help='The model to be optimized.')
parser.add_argument('--nesterov',
                    help='Use nesterov SGD', action='store_true',
                    default=False)                    
parser.add_argument('--description',
                    help='Description for overall experiment', type=str,
                    default="None")
parser.add_argument('--configpath', default=None,
                        help='Configuration path for decomposition JSON file')
#parser.add_argument('--layer',default=3, type=int, help='layer to replace in network' )
#parser.add_argument('--rank', default = 448, type=int, help='Decomposition rank')
parser.add_argument('--mode', choices=['None', 'CP', 'Weight', 'Tai'], default='None',
                        help='Required normalization mode')
parser.add_argument('--optimizer', choices=['SGD', 'RMSPROP'], default='SGD',
                        help='Optimizer to use')
                        
parser.add_argument('--compress_rate', type=float, default=0, metavar='N',
                    help='Compression rate for the network compression')
parser.set_defaults(augment=True)


best_prec1 = 0
args = parser.parse_args()

print(args, flush=True)

print("Tensorboard: ",args.tensorboard, flush=True)


if args.tensorboard:
    writer = SummaryWriter(comment='_' + args.data_set + '_'
                                   + '_lr_' + str(args.lr)
                                   + '_m_' + str(args.momentum)+ '_'
                                   + '_mode_' + args.mode + '_'
                                   + '_optim_'+ args.optimizer + '_'
                                   + '_compress-rate_' + str(args.compress_rate))


def main():
    global args, best_prec1
    #torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(args.seed)
    # Data loading code
    # to_tensor transform includes division by 255.
    # see https://pytorch.org/docs/stable/torchvision/
    # transforms.html#torchvision.transforms.ToTensor
    if args.data_set == 'cifar10':
        
        normalize = transforms.Normalize(
            mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
            std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    elif args.data_set == 'SVHN':
        normalize = transforms.Normalize(
            mean=[x/255.0 for x in [111.6, 113.2, 120.6]],
            std=[x/255.0 for x in [50.5, 51.3, 50.2]])
    else:
        raise ValueError('Unkown data set.')

    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])


    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = None
    val_loader = None
    # CIFAR10 dataset
    if args.data_set == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=True, download=True,
                            transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    # SVHN dataset
    elif args.data_set == 'SVHN':
        train_loader = torch.utils.data.DataLoader(
           datasets.SVHN('./svhn_data', split='train', download=True,
                         transform=transform_train),
                         batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.SVHN('./svhn_data', split='test', download=True,
                         transform=transform_test),
                         batch_size=args.batch_size, shuffle=True, **kwargs)
    else:
        print('Unknown dataset', flush=True)
    # model instantiation
    if args.model == 'AlexNet':
        model = AlexNet()
    else:
        raise ValueError('Unkown model.')

    # get the number of model parameters
    print('Number of model parameters before: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])), flush=True)

    if args.resume:
        if os.path.isfile(args.name):
            print("=> loading checkpoint '{}'".format(args.name), flush=True)
            checkpoint = torch.load(args.name)
            args.start_epoch = checkpoint['epoch']
            args.epochs += args.start_epoch
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.name, checkpoint['epoch']), flush=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.name), flush=True)

    # Apply CP norm based on inference/training mode
    if args.mode == 'CP':
        print("Applying CP Norm", flush=True)
        print(args.resume)
        if args.resume:
            print('inference/fine tuning mode', flush=True)
            model = apply_CP_Norm(model, True)
        else:
            print('training mode', flush=True)
            model = apply_CP_Norm(model)
        print()
        print("CP Norm application done", flush=True)    
    # Apply weight norm    
    elif args.mode == 'Weight':
        print('Applying weight norm', flush=True)
        model = apply_Weight_Norm(model)
        print()
        print('Weight norm application done', flush=True)
    elif args.mode == 'Tai':
        print()
        print(model.eval())
        model = tai_decompose(model, args.configpath)
        print()
        print(model.eval(), flush=True)
    # for training on multiple GPUs. 
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint

    if not args.cpu:
        model = model.cuda()


    cudnn.benchmark = True

    # Compression code
    if args.mode == 'CP' and args.compress_rate != 0:
        print('Running compression.....', flush=True)
        model = apply_compression(model, args.compress_rate)
     
    # define loss function (criterion) and optimizer
    if not args.cpu:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'SGD':
        print('Momentum: ', args.momentum)
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=args.nesterov)
    elif args.optimizer == 'RMSPROP':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(),
    #                              args.lr,
    #                              momentum=args.momentum,
    #                              weight_decay=args.weight_decay)
    print('Number of model parameters after: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])), flush=True)
    no_params = sum([p.data.nelement() for p in model.parameters()])
    writer.add_scalar('Parameters', no_params, 0)
    print('using :', optimizer)
    print("Afer optim")
    
    # Training loop
    wait = 0
    patience = 8
    min_val_loss = np.Inf
    for epoch in range(args.start_epoch, args.epochs):
        print("Epoch: ", epoch)
        # adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        # model = apply_Weight_Norm(model)
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1, val_loss = validate(val_loader, model, criterion, epoch)


        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)

        ## Early stopping code
        #if val_loss<= min_val_loss:
        #    print('Loss decreased', flush=True)
        #    wait = 0
        #    min_val_loss = val_loss
        #else:
        #    wait += 1
        #    if wait >= patience:
        #        print('Terminating becuase of early stopping', flush=True)
        #        break
    print('Best accuracy: ', best_prec1)

def apply_CP_Norm(model, inference_type = False):
    """
    Method to apply CP Normalization over the required layers. In this 
    case the whole network

    Args:
        model (Net): Model instance [Net] 
        inference_type (bool, optional): Boolean variable for inference 
                                         or trianing mode. Defaults to False.

    Returns:
        [Net]: CP norm applied model instance
    """
    # Ranks for convolutional layers
    ranks = [36, 571, 1626, 1948, 1644]
    model = model.cpu()
    c = 0
    # Apply CPnorm over convolutional layers
    for index, (name, layer) in enumerate(model.named_modules()):
        if isinstance(layer, nn.Conv2d):
            print('Norm on layer: ', name)
            layer = cp_norm(layer, ranks[c], inference=inference_type)
            c+=1
    # Ranks for linear layers
    linear_ranks = [1024, 512, 10]
    c = 0
    # Apply CPnorm over linear layers
    for index, (name, layer) in enumerate(model.named_modules()):
        if isinstance(layer, nn.Linear):
            layer = cp_norm(layer, linear_ranks[c], inference=inference_type)
            c+=1
    model = model.cuda()
    return model

def apply_Weight_Norm(model):
    """
    Method to apply weight norm over required layers. In this case the
    whole network

    Args:
        model (Net): Model instance [Net] 

    Returns:
        [Net]: Weight norm applied model instance
    """
    for index, (name, layer) in enumerate(model.named_modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer = torch.nn.utils.weight_norm(layer)
    return model

def train(train_loader, model, criterion, optimizer, epoch):
    """
    Method to train for one epoch on training set

    Args:
        train_loader (Dataloader): Dataloader object of training examples
        model (Net): Model instance [Net] 
        criterion (Criterion): Loss function for the training loop
        optimizer (Optimizer): Specified optimizer for the training
        epoch (int): Current epoch value
    """
    batch_time = AverageMeter()
    closses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if not args.cpu:
            target = target.cuda()
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        closs = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        closses.update(closs.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        closs.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=closses, top1=top1))
    
    # log to TensorBoard
    if args.tensorboard:
        #log_value('train_loss', losses.avg, epoch)
        writer.add_scalar('train_loss', closses.avg, epoch)
        # log_value('train_acc', top1.avg, epoch)
        writer.add_scalar('train_acc', top1.avg, epoch)
        

def validate(val_loader, model, criterion, epoch):
    """
    Method to perform validation on validation set

    Args:
        val_loader (Datalaoder): Dataloader object with validation examples
        model (Net): Model instance [Net] 
        criterion (Criterion): Loss function for the training loop
        epoch (int): Current epoch value

    Returns:
        [float]: top 1% accuracy average value
        [float]: Validation loss average value
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if not args.cpu:
            target = target.cuda(non_blocking=True)
            input = input.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1), flush=True)
    # log to TensorBoard
    if args.tensorboard:
        # log_value('val_loss', losses.avg, epoch)
        writer.add_scalar('val_loss', losses.avg, epoch)
        # log_value('val_acc', top1.avg, epoch)
        writer.add_scalar('val_acc', top1.avg, epoch)
    return top1.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Method to save the checkpoint to disk

    Args:
        state (dict): A dictionary containing the state values
        is_best (bool): Boolean value to specify best model
        filename (str, optional): Fine name. Defaults to 'checkpoint.pth.tar'.
    """
    directory = "test_runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'test_runs/%s/'%(args.name) + 
                        'model_best.pth.tar')

class AverageMeter(object):
    """
    Method to compute the average and current value

    Args:

    """
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    # log to TensorBoard
    if args.tensorboard:
        # log_value('learning_rate', lr, epoch)
        writer.add_scalar('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k

    Args:
        output (Tensor): Tensor containing the predicted classes
        target (Tensor): Tensor containing the labels
        topk (tuple, optional): Calculate particular k. Defaults to (1,).

    Returns:
        [float]: top k precision calculated value
    """
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
