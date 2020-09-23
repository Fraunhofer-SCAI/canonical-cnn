# -*- coding: utf-8 -*-
"""Alexnet_Implementation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_t3bY-LWkvyKkCdVhdyoCFEHP1C_9EY6

The implementation is completely implemented from https://analyticsindiamag.com/implementing-alexnet-using-pytorch-as-a-transfer-learning-model-in-multi-class-classification/
"""
from collections import OrderedDict
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from CP_N_Way_Decomposition import CP_ALS
from tensorly.decomposition import parafac
import tensorly as tl

print("### Libraries loaded and locked", flush=True)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#Downloading training data
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)

#Downloading test data
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=2)

#Class labels

classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
print("### Dataset loaded and locked", flush=True)
#import matplotlib.pyplot as plt
#import numpy as np

#Function to show some random images
#def imshow(img):
#    img = img / 2 + 0.5     # unnormalize
#    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#    plt.show()

#Get some random training images
#dataiter = iter(trainloader)
#images, labels = dataiter.next()

#Show images
#imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

#Now using the AlexNet
AlexNet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
print("### Alexnet model loaded and locked", flush=True)
#Model description
print("Before changing", flush=True)
print(AlexNet_model.eval(), flush=True)
# Retreive the second layer conv2d weight tensor
second_weight_tensor = AlexNet_model.features[3].weight.data
# Compute CP decomposition of this tensor
# Parameters of the decomposition
max_iter = 100
r = 64
r_state = 0
cp = CP_ALS()
print("Computing the factors of the weight tensor", flush=True)
start = time.time()
#A, lmbds = cp.compute_ALS(second_weight_tensor, max_iter, rank)
x = parafac(tl.tensor(second_weight_tensor), rank=r, normalize_factors=False, init='random', random_state=r_state, n_iter_max= max_iter)
end = time.time()
A = x[1]
print("Factors calculated in ", end-start," seconds", flush=True)
print("Factors shapes are: ", flush=True)
#K_t, K_s, K_y, K_x = A[0], A[1], A[2], A[3]
K_t, K_s, K_y, K_x = torch.from_numpy(A[0]), torch.from_numpy(A[1]), torch.from_numpy(A[2]), torch.from_numpy(A[3])
print((K_s.T.unsqueeze(-1).unsqueeze(-1)).shape, flush=True)
print((K_y.T.unsqueeze(1).unsqueeze(1)).shape, flush=True)
print((K_x.T.unsqueeze(0).unsqueeze(-1)).shape, flush=True)
print((K_t.unsqueeze(-1).unsqueeze(-1)).shape, flush=True)
K_s = K_s.T.unsqueeze(-1).unsqueeze(-1)
K_y = K_y.T.unsqueeze(1).unsqueeze(0)
K_x = K_x.T.unsqueeze(1).unsqueeze(-1)
K_t = K_t.unsqueeze(-1).unsqueeze(-1)
model = nn.Sequential(OrderedDict([
          ('K_s', torch.nn.Conv2d(K_s.shape[0], K_s.shape[1], kernel_size = (K_s.shape[2], K_s.shape[3]), stride = (1, 1), padding = (2, 2), bias=False)),
          ('K_y', torch.nn.Conv2d(K_y.shape[0], K_y.shape[1], kernel_size = (K_y.shape[2], K_y.shape[3]), stride = (1, 1), padding = (2, 2), bias=False)),
          ('K_x', torch.nn.Conv2d(K_x.shape[0], K_x.shape[1], kernel_size = (K_x.shape[2], K_x.shape[3]), stride = (1, 1), padding = (2, 2), bias = False)),
          ('K_t', torch.nn.Conv2d(K_t.shape[0], K_t.shape[1], kernel_size = (K_t.shape[2], K_t.shape[3]), stride = (1, 1), padding = (2, 2), bias=False))
        ]))
AlexNet_model.features[3] = model
print("ABC")
print(AlexNet_model.features[3][0], flush=True)
print("Loading the values: ")
with torch.no_grad():
    AlexNet_model.features[3][0].weight = torch.nn.Parameter(K_s)
    AlexNet_model.features[3][1].weight = torch.nn.Parameter(K_y)
    AlexNet_model.features[3][2].weight = torch.nn.Parameter(K_x)
    AlexNet_model.features[3][3].weight = torch.nn.Parameter(K_t)
print("Values are loaded")
#Updating the second classifier
AlexNet_model.classifier[4] = torch.nn.Linear(4096,1024)

#Updating the third and the last classifier that is the output layer of the network. Make sure to have 10 output nodes if we are going to get 10 class labels through our model.
AlexNet_model.classifier[6] = torch.nn.Linear(1024,10)
print("After changing", flush=True)
print(AlexNet_model.eval(), flush=True)

#Instantiating CUDA device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Verifying CUDA
print(device, flush=True)

#Move the input and AlexNet_model to GPU for speed if available
AlexNet_model.to(device)

#Loss
criterion = nn.CrossEntropyLoss()

#Optimizer(SGD)
optimizer = optim.SGD(AlexNet_model.parameters(), lr=0.001, momentum=0.01)
print("### Optimizer loaded and locked", flush=True)
print("### Training started ", flush=True)

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)#

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = AlexNet_model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training of AlexNet', flush=True)

#Testing Accuracy
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = AlexNet_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

#Testing classification accuracy for individual classes.
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = AlexNet_model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

avg = 0
for i in range(10):
  temp = (100 * class_correct[i] / class_total[i])
  avg = avg + temp
avg = avg/10
print('Average accuracy = ', avg)