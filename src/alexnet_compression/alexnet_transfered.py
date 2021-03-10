import argparse
import os
import time
from os import cpu_count
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
import numpy as np
import random
tl.set_backend("pytorch")
from tensorly.decomposition import parafac
from cp_norm import cp_norm

parser = argparse.ArgumentParser(description='PyTorch Cifar-AlexNet Training')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.6, type=float, help='momentum')
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
parser.add_argument('--layer',default=3, type=int, help='layer to replace in network' )
parser.add_argument('--rank', default = 448, type=int, help='Decomposition rank')
parser.add_argument('--mode', default=0, type=int, help='If 0 then normal, else if 1 then cpnorm or else\
                    2 then weight norm')
parser.add_argument('--optimizer', default=0, type=int, help='Select an optimizer\
                    If 0 then SGD else 1 it is RMSProp')
parser.add_argument('--compress_rate', type=int, default=0, metavar='N',
                    help='Compression rate for the network compression')
parser.set_defaults(augment=True)


best_prec1 = 0
args = parser.parse_args()

#args.resume = "./runs/exp1/checkpoint.pth.tar"
args.resume = './exp1_kernel/checkpoint.pth.tar'
#args.weight_decay = 0.1
print(args, flush=True)

print("Tensorboard: ",args.tensorboard, flush=True)
status = None
if args.mode == 0:
    status = 'None'
elif args.mode == 1:
    status = 'CPNorm'
elif args.mode == 2:
    status = 'WeightNorm'
used_optim = None
if args.optimizer == 0:
    used_optim = 'SGD'
elif args.optimizer == 1:
    used_optim = 'RMSProp'
if args.tensorboard:
    writer = SummaryWriter(comment='_' + args.data_set + '_'
                                   + '_lr_' + str(args.lr)
                                   + '_m_' + str(args.momentum)+ '_'
                                   + '_rank_' + str(args.rank)+ '_'
                                   + '_mode_' + status + '_'
                                   + '_optim_'+ used_optim + '_'
                                   + '_wdecay_' + str(args.weight_decay))


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
    print('Number of model parameters before: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])), flush=True)
    
    if args.mode == 1:
        print("Applying CP Norm", flush=True)
        if os.path.isfile(args.resume):
            model = apply_CP_Norm(model, True)
        else:
            model = apply_CP_Norm(model)
        print()
        print("CP Norm application done", flush=True)        
    elif args.mode == 2:
        print('Applying weight norm', flush=True)
        model = apply_Weight_Norm(model)
        print()
        print('Weight norm application done', flush=True)
    # for training on multiple GPUs. 
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    if not args.cpu:
        model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume), flush=True)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']), flush=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume), flush=True)

    cudnn.benchmark = True

    # Compression code
    if args.mode == 1 and args.compress_rate != 0:
        print('Running compression.....', flush=True)
        model = apply_compression(model, args.compress_rate)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    for index, (name, lay) in enumerate(model.named_modules()):
        if index in [5, 6, 8]:
            lay.register_forward_hook(get_activation(name))
     
    # define loss function (criterion) and optimizer
    if not args.cpu:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    if args.optimizer == 0:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=args.nesterov)
    elif args.optimizer == 1:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(),
    #                              args.lr,
    #                              momentum=args.momentum,
    #                              weight_decay=args.weight_decay)
    print('Number of model parameters after: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])), flush=True)

    print('using :', optimizer)
    print("Afer optim")
    for epoch in range(args.start_epoch, args.epochs):
        print("Epoch: ", epoch)
        #adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        # model = apply_Weight_Norm(model)
        train(train_loader, model, criterion, optimizer, epoch, activation)

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

def apply_CP_Norm(model, inference_type):#
    ranks = [40, 571, 1540, 1800, 1482]
    model = model.cpu()
    c = 0
    for index, (name, layer) in enumerate(model.named_modules()):
        if isinstance(layer, nn.Conv2d):
            layer = cp_norm(layer, ranks[c], inference=inference_type)
            c+=1
    model = model.cuda()
    return model

def apply_Weight_Norm(model):
    for index, (name, layer) in enumerate(model.named_modules()):
        if isinstance(layer, nn.Conv2d):# or isinstance(layer, nn.Linear):
            layer = torch.nn.utils.weight_norm(layer)
    return model

def train(train_loader, model, criterion, optimizer, epoch, activation):
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
    # Get activation maps between the layers and unfold them
    # Calculate the mean norm and flatten norm
    mean_norm = []
    flatten_norm = []
    for index, key in enumerate(activation):
        inp = activation[key]
        kernel = (1, 1)
        if index == 2:
            kernel == (3, 3)
        inp_unf = torch.nn.functional.unfold(inp, kernel)
        norms = []
        for index in range(0, inp_unf.shape[0]):
            norms.append(np.linalg.norm(inp_unf[index, :, :].cpu().detach().numpy(), ord=2))
        mean_norm.append(sum(norms)/len(norms))
        flatten_norm.append(np.linalg.norm(inp_unf.flatten().cpu().detach().numpy(), ord=2))
    weight_norms = []
    for index, (n, p) in enumerate(model.named_parameters()):
        if index in [2, 3, 4, 5]:
            print(p.size())
            p_unfolded = p.view(p.size(0), -1)
            weight_norms.append(np.linalg.norm(p_unfolded.cpu().detach().numpy(), ord=2))


    

    # log to TensorBoard
    if args.tensorboard:
        #log_value('train_loss', losses.avg, epoch)
        writer.add_scalar('train_loss', closses.avg, epoch)
        # log_value('train_acc', top1.avg, epoch)
        writer.add_scalar('train_acc', top1.avg, epoch)

        writer.add_scalar("Layer 2 input mean norm", mean_norm[0], epoch)
        writer.add_scalar("Layer 3 input mean norm", mean_norm[1], epoch)
        writer.add_scalar("Layer 4 input mean norm", mean_norm[2], epoch) 
        
        writer.add_scalar("Layer 2 input flatten norm", flatten_norm[0], epoch) 
        writer.add_scalar("Layer 3 input flatten norm", flatten_norm[1], epoch) 
        writer.add_scalar("Layer 4 input flatten norm", flatten_norm[2], epoch) 

        writer.add_scalar("Layer 2 weight norm", weight_norms[0], epoch)
        writer.add_scalar("Layer 3 weight norm", weight_norms[1], epoch)
        writer.add_scalar("Layer 4 weight norm", weight_norms[2], epoch)
        

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

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1), flush=True)
    # log to TensorBoard
    if args.tensorboard:
        # log_value('val_loss', losses.avg, epoch)
        writer.add_scalar('val_loss', losses.avg, epoch)
        # log_value('val_acc', top1.avg, epoch)
        writer.add_scalar('val_acc', top1.avg, epoch)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "test_runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'test_runs/%s/'%(args.name) + 'model_best.pth.tar')

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
    # weight_tensor = AlexNet_model.features[layer].weight.data
    g_weight = AlexNet_model.features[layer].weight_g.data
    v_weight = AlexNet_model.features[layer].weight_v.data
    weight_tensor = g_weight * v_weight
    # weight_tensor = v_weight
    max_iter = 200
    cp = CP_ALS()
    
    ## Implementation from https://github.com/ruihangdu/Decompose-CNN/blob/master/scripts/torch_cp_decomp.py

    # last, first, vertical, horizontal = parafac(weight_tensor, rank=args.rank, init='random', n_iter_max = max_iter)[1]
    last, first, vertical, horizontal = cp.compute_ALS(weight_tensor, max_iter, args.rank)[0]
    # Uncomment the below two lines in case of normalized CP Decomposition
    # factors, lmbds = cp.compute_ALS(weight_tensor, max_iter, args.rank, norms=True)
    # last, first, vertical, horizontal = factors[0], factors[1], factors[2], factors[3]
        
    pointwise_s_to_r_layer = nn.Conv2d(in_channels=first.shape[0],
                                       out_channels=first.shape[1],
                                       kernel_size=1,
                                       padding=0,
                                       bias=False)

    depthwise_r_to_r_layer = nn.Conv2d(in_channels=args.rank,
                                       out_channels=args.rank,
                                       kernel_size=vertical.shape[0],
                                       stride=AlexNet_model.features[layer].stride,
                                       padding=AlexNet_model.features[layer].padding,
                                       dilation=AlexNet_model.features[layer].dilation,
                                       groups=args.rank,
                                       bias=False)
                                       
    pointwise_r_to_t_layer = nn.Conv2d(in_channels=last.shape[1],
                                       out_channels=last.shape[0],
                                       kernel_size=1,
                                       padding=0,
                                       bias=True)
    
    if AlexNet_model.features[layer].bias is not None:
        pointwise_r_to_t_layer.bias.data = AlexNet_model.features[layer].bias.data
    
    sr = first.t_().unsqueeze_(-1).unsqueeze_(-1)
    rt = last.unsqueeze_(-1).unsqueeze_(-1)
    rr = torch.stack([vertical.narrow(1, i, 1) @ torch.t(horizontal).narrow(0, i, 1) for i in range(args.rank)]).unsqueeze_(1)
    pointwise_s_to_r_layer.weight.data = sr 
    pointwise_r_to_t_layer.weight.data = rt
    depthwise_r_to_r_layer.weight.data = rr
    model = [pointwise_s_to_r_layer, depthwise_r_to_r_layer, pointwise_r_to_t_layer]

    AlexNet_model.features = nn.Sequential(\
        *(list(AlexNet_model.features[:layer]) + model + list(AlexNet_model.features[layer + 1:])))
    if not args.cpu:
        AlexNet_model = AlexNet_model.cuda()
    return AlexNet_model

def compress_via_reparam(layer, compress_rate):
    if isinstance(layer, torch.nn.Conv2d):
        lmbds = layer.weight_weights
        k = int(len(lmbds)*(1-(compress_rate/100)))
        print('k value: ',k, flush=True)
        A, B = layer.weight_A, layer.weight_B
        C, D = layer.weight_C, layer.weight_D
        lmbds_sorted, indices = torch.sort(lmbds, descending=True)
        lmbds_sorted, indices = lmbds_sorted[0:k], indices[0:k]
        A, B, C, D = A[:, indices], B[:, indices], C[:, indices], D[:, indices]
        layer.weight_weights = torch.nn.Parameter(lmbds_sorted)
        layer.weight_A, layer.weight_B = torch.nn.Parameter(A), torch.nn.Parameter(B)
        layer.weight_C, layer.weight_D = torch.nn.Parameter(C), torch.nn.Parameter(D)

    if isinstance(layer, torch.nn.Linear):
        lmbds = layer.weight_weights
        k = int(len(lmbds)*(1-(compress_rate/100)))
        print('k value: ',k, flush=True)
        A, B = layer.weight_A, layer.weight_B
        lmbds_sorted, indices = torch.sort(lmbds, descending=True)
        lmbds_sorted, indices = lmbds_sorted[0:k], indices[0:k]
        A, B = A[:, indices], B[:, indices]
        layer.weight_weights = torch.nn.Parameter(lmbds_sorted)
        layer.weight_A, layer.weight_B = torch.nn.Parameter(A), torch.nn.Parameter(B)
    
    return layer


def apply_compression(model, compress_rate):
    for index, (name, layer) in enumerate(model.named_modules()):
        if isinstance(layer, torch.nn.Conv2d):
            layer = compress_via_reparam(layer, compress_rate)
    return model

if __name__ == '__main__':
    main()
