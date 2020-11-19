import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from alexnet_model import AlexNet
from CP_N_Way_Decomposition import CP_ALS
from collections import OrderedDict
import tensorly as tl
tl.set_backend("pytorch")
from tensorly.decomposition import parafac

parser = argparse.ArgumentParser(description='PyTorch Cifar-AlexNet Training')
parser.add_argument('--epochs', default=310, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.1, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0., type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='exp1_kernel', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true',
                    default=True)
parser.add_argument('--cpu', help='Run on cpu only', action='store_true',
                    default=False)
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
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
parser.add_argument('--layer',default=3, type=int, help='layer to replace in network' )
parser.add_argument('--rank', default = 448, type=int, help='Decomposition rank')

parser.set_defaults(augment=True)


best_prec1 = 0
args = parser.parse_args()

args.description = 'Kernel_Compression_Test_1_Weight_Norms_Tensorly_PAckahe'
args.resume = "./runs/exp1/checkpoint.pth.tar"
print(args)

print("Tensorboard: ",args.tensorboard)
if args.tensorboard:
    writer = SummaryWriter(comment='_' + args.data_set + '_'
                                   + '_lr_' + str(args.lr)
                                   + '_m_' + str(args.momentum)+ '_'
                                   + args.description)


def main():
    global args, best_prec1

    torch.manual_seed(args.seed)
    # Data loading code
    # to_tensor transform includes division by 255.
    # see https://pytorch.org/docs/stable/torchvision/
    # transforms.html#torchvision.transforms.ToTensor
    if args.data_set == 'cifar10':
        
        normalize = transforms.Normalize(
            mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
            std=[x/255.0 for x in [63.0, 62.1, 66.7]])
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
    if args.data_set == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=True, download=True,
                            transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    # create model
    if args.model == 'AlexNet':
        model = AlexNet()
    else:
        raise ValueError('Unkown model.')
    # print(model)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    
    # for training on multiple GPUs. 
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    if not args.cpu:
        model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    print()
    print("Before replacement: ")
    print(model.eval())
    print()
    model = replace_layer(model, args.layer)

    print()
    print("After replacement: ")
    print(model.eval())
    print()
    # define loss function (criterion) and pptimizer
    if not args.cpu:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    for i, (name, param) in enumerate(model.named_parameters()):
        if "features.3" not in name:
            param.requires_grad=False
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    # optimizer = torch.optim.Adam(model.parameters(),
    #                              args.lr,
    #                              momentum=args.momentum,
    #                              weight_decay=args.weight_decay)
    for param_group in optimizer.param_groups:
        print("Learning rate: ", param_group['lr'])
    print('using :', optimizer)
    print("Afer optim")
    for epoch in range(args.start_epoch, args.epochs):
        print("Epoch: ", epoch)
        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('Best accuracy: ', best_prec1)


def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
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
    norms = []
    for i, (name, param) in enumerate(model.named_parameters()):
        if "features.3"  in name:
            norms.append(torch.norm(param))
    #print("Norms: ")
    #print(norms)
    #print()
    # log to TensorBoard
    if args.tensorboard:
        #log_value('train_loss', losses.avg, epoch)
        writer.add_scalar('train_loss', closses.avg, epoch)
        # log_value('train_acc', top1.avg, epoch)
        writer.add_scalar('train_acc', top1.avg, epoch)
        writer.add_scalar('K_s norm', norms[0], epoch)
        writer.add_scalar('K_y norm', norms[1], epoch)
        writer.add_scalar('K_x norm', norms[2], epoch)
        writer.add_scalar('K_t norm', norms[3], epoch)


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
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

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        # log_value('val_loss', losses.avg, epoch)
        writer.add_scalar('val_loss', losses.avg, epoch)
        # log_value('val_acc', top1.avg, epoch)
        writer.add_scalar('val_acc', top1.avg, epoch)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

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



def replace_layer(AlexNet_model, layer):
    AlexNet_model = AlexNet_model.cpu()
    weight_tensor = AlexNet_model.features[layer].weight.data
    max_iter = 200
    cp = CP_ALS()
    print()
    print("Rank is : ", args.rank)
    print("Computing the factors of the weight tensor", flush=True)
    start = time.time()
    #A, lmbds = cp.compute_ALS(weight_tensor, max_iter, args.rank)
    A = parafac(weight_tensor, args.rank, n_iter_max = max_iter, init="random", normalize_factors=True)[1]
    end = time.time()
    print("Factors calculated in ", end-start," seconds", flush=True)
    K_t, K_s, K_y, K_x = A[0], A[1], A[2], A[3]
    print(K_t.shape, K_s.shape, K_y.shape, K_x.shape)
    K_s_a = K_s.T.unsqueeze(-1).unsqueeze(-1)
    K_y_a = K_y.T.unsqueeze(1).unsqueeze(0)
    K_x_a = K_x.T.unsqueeze(1).unsqueeze(-1)
    K_t_a = K_t.unsqueeze(-1).unsqueeze(-1)
    K_s = K_s.unsqueeze(-1).unsqueeze(-1)
    K_y = K_y.T.unsqueeze(1).unsqueeze(1)
    K_x = K_x.T.unsqueeze(0).unsqueeze(-1)
    K_t = K_t.T.unsqueeze(-1).unsqueeze(-1)
    model = torch.nn.Sequential(OrderedDict([
            ('K_s', torch.nn.Conv2d(K_s.shape[0], K_s.shape[1], kernel_size = (K_s.shape[2], K_s.shape[3]),  padding = (0, 0))),
            ('K_y', torch.nn.Conv2d(K_y.shape[0], K_y.shape[1], kernel_size = (K_y.shape[2], K_y.shape[3]),  padding = (0, 0))),
            ('K_x', torch.nn.Conv2d(K_x.shape[0], K_x.shape[1], kernel_size = (K_x.shape[2], K_x.shape[3]),  padding = (0, 0))),
            ('K_t', torch.nn.Conv2d(K_t.shape[0], K_t.shape[1], kernel_size = (K_t.shape[2], K_t.shape[3]),  padding = (1, 1)))
            ]))
    AlexNet_model.features[layer] = model
    #with torch.no_grad():
    print("Layer shape: ")
    print(K_s.shape, K_y.shape, K_x.shape, K_t.shape)
    print("Input shape: ")
    print(K_s_a.shape, K_y_a.shape, K_x_a.shape, K_t_a.shape)
    AlexNet_model.features[layer][0].weight = torch.nn.Parameter(K_s_a)
    AlexNet_model.features[layer][1].weight = torch.nn.Parameter(K_y_a)
    AlexNet_model.features[layer][2].weight = torch.nn.Parameter(K_x_a)
    AlexNet_model.features[layer][3].weight = torch.nn.Parameter(K_t_a)
    return AlexNet_model.cuda()

if __name__ == '__main__':
    main()