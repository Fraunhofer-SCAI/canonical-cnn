# ====================================================================
# Train LeNet like architecture
# ====================================================================

from __future__ import print_function, absolute_import
import argparse
from typing import ValuesView
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets, transforms

from src.cp_compress import apply_compression
from models.ConvNet_model import Net


def compute_parameter_total(net):
    """
    Method to compute the total number of parameters in the model

    Args:
     Net [model]: Model instance to compute total parameters
    
    Returns:
     [int]: Total number of paramters for the instantiated model
    """
    total = 0
    for p in net.parameters():
        if p.requires_grad:
            print(p.shape)
            total += np.prod(p.shape)
    return total


def train(args, model, device, train_loader, optimizer, epoch, writer):
    """
    Method to train the model with corresponding dataloader & write 
    result to tensorboard session

    Args:
        args (parser): Argument parser holding all hyperparameters
        model (Net): Instance of model [Net] 
        device (str): String holding type of device [cpu or cuda]
        train_loader (dataloader): Dataloader for trainig images
        optimizer (optimizer): Specified optimizer
        epoch (int): Current training epoch
        writer (SummaryWriter): Summary writer object for tensorboard
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # Forward pass
        output = model(data)
        # Loss calculation
        loss = F.nll_loss(output, target)
        loss.backward()
        # Backward pass
        optimizer.step()
        # Printing and tensorboard writer
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), flush=True)
            if args.dry_run:
                break
        writer.add_scalar('train_loss', loss.item(), epoch)


def test(model, device, test_loader, epoch, writer):
    """
    Method to test the model with corresponding dataloader & write 
    result to tensorboard session

    Args:
        model (Net): Instance of model [Net]
        device (str): String holding type of device [cpu or cuda]
        test_loader (dataloader): Dataloader for testing images
        epoch (int): Current training epoch
        writer (SummaryWriter): Summary writer object for tensorboard
    """
    model.eval()
    test_loss = 0
    correct = 0
    # Forward pass on testing images
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Accuracy calculation
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    # Printing and tensorboard writing
    writer.add_scalar('test_loss', test_loss, epoch)
    writer.add_scalar('test_acc', accuracy, epoch)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)), flush=True)


def main():
    torch.autograd.set_detect_anomaly(True)
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
    parser.add_argument('--mode', choices=['None', 'CP', 'Weight'], default='None',
                        help='Required normalization mode')
    parser.add_argument('--optimizer', choices=['SGD', 'RMSPROP'], default='SGD',
                        help='Optimizer to use')
    parser.add_argument('--compress_rate', type=int, default=0, metavar='N',
                        help='Compression rate for the network compression')
    parser.add_argument('--name', default='./mnist_cnn_rmsprop.pt', 
                        help='Name to apply saved weights')
    


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    writer = SummaryWriter(comment='_' + 'MNIST' + '_'
                                   + '_lr_' + str(args.lr) + '_'
                                   + '_mode_'+ args.mode + '_'
                                   + '_optim_'+ args.optimizer+ '_'
                                   + '_compress-rate_' + 
                                   str(args.compress_rate))
    torch.manual_seed(args.seed)
    print(args)

    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('deivce: ', device, flush=True)
    # Dataloader creation
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Model instantiation
    net_model = None
    if args.mode == 'None':
        net_model = Net().to(device)
    elif args.mode == 'CP':
        net_model = Net(cpnorm=True).to(device)
    elif args.mode == 'Weight':
        net_model = Net(wnorm=True).to(device)
    parameter_total = compute_parameter_total(net_model)
    print('new parameter total ###:', parameter_total, flush=True)
    if args.compress_rate != 0:
        writer.add_scalar('params', parameter_total, 0)

    
    # Optimizer creation
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(net_model.parameters(), lr=args.lr)
    elif args.optimizer == 'RMSPROP':
        optimizer = optim.RMSprop(net_model.parameters(), lr=args.lr)

    if args.resume:
        net_model.load_state_dict(torch.load(args.name))

    # Compression code
    if args.mode == 'CP' and args.compress_rate != 0:
        print('Running compression.....', flush=True)
        net_model = apply_compression(net_model, args.compress_rate)
        new_parameter_total = compute_parameter_total(net_model)
        print('Compression parameter total:', new_parameter_total)
        writer.add_scalar('params', new_parameter_total, 1)
    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # Trainig loop
    for epoch in range(1, args.epochs+1):
        train(args, net_model, device, train_loader, optimizer, epoch, writer)
        test(net_model, device, test_loader, epoch, writer)
        #scheduler.step()

    # Save parameters in tensorboard and save the model
    writer.close()
    if args.save_model:
        fname = './mnist_cnn_crate-'+str(args.compress_rate)+'.pt'
        torch.save(net_model.state_dict(), fname)



if __name__ == '__main__':
    main()