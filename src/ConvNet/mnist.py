from __future__ import print_function
import argparse
from typing import ValuesView

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets, transforms

from ...cp_compress import apply_compression
from ...models.ConvNet_model import Net


def compute_parameter_total(net):
    total = 0
    for p in net.parameters():
        if p.requires_grad:
            print(p.shape)
            total += np.prod(p.shape)
    return total


def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        writer.add_scalar('train_loss', loss.item(), epoch)


def test(model, device, test_loader, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    writer.add_scalar('test_loss', test_loss, epoch)
    writer.add_scalar('test_acc', accuracy, epoch)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=int(time.time()), metavar='S',
                        help='random seed (default: int(time.time()))')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training from a stored model.')
    parser.add_argument('--mode', type=int, default=0, metavar='N', 
                        help ='0 for normal, 1 for CPnorm and 2 for weightnorm')
    parser.add_argument('--optimizer', type=int, default=0, metavar='N',
                        help='0 for SGD, 1 for RMSProp')


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    status = ''
    if args.mode == 0:
        status = 'None'
    elif args.mode == 1:
        status = 'CP-Norm'
    elif args.mode == 2:
        status = 'Weight-Norm'
    opti = None
    if args.optimizer == 0:
        opti = 'SGD'
    elif args.optimizer == 1:
        opti = 'RMSProp'
    writer = SummaryWriter(comment='_' + 'MNIST' + '_'
                                   + '_lr_' + str(args.lr) + '_'
                                   + '_mode_'+ status + '_'
                                   + '_optim_'+ opti)
    torch.manual_seed(args.seed)
    print(args)

    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('deivce: ', device, flush=True)
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    net_model = None
    if args.mode == 0:
        net_model = Net().to(device)
    elif args.mode == 1:
        net_model = Net(cpnorm=True).to(device)
    elif args.mode == 2:
        net_model = Net(wnorm=True).to(device)
    parameter_total = compute_parameter_total(net_model)
    print('new parameter total:', parameter_total)

    if args.optimizer == 0:
        optimizer = optim.SGD(net_model.parameters(), lr=args.lr)#, momentum=0.1)#, weight_decay = 0.1)
    elif args.optimizer == 1:
        optimizer = optim.RMSprop(net_model.parameters(), lr=args.lr)

    if args.resume:
        net_model.load_state_dict(torch.load("./mnist_cnn.pt"))

    # Compression code
    if args.mode == 1 and args.compress_rate != 0:
        print('Running compression.....', flush=True)
        net_model = apply_compression(net_model, args.compress_rate)
    parameter_total = compute_parameter_total(net_model)
    print('Compression parameter total:', parameter_total)
    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs+1):
        train(args, net_model, device, train_loader, optimizer, epoch, writer)
        test(net_model, device, test_loader, epoch, writer)
        #scheduler.step()

    writer.close()
    if args.save_model:
        torch.save(net_model.state_dict(), "./mnist_cnn.pt")



if __name__ == '__main__':
    main()