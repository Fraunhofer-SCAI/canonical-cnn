# -*- coding: utf-8 -*-
from collections import OrderedDict
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from CP_N_Way_Decomposition import CP_ALS
#from tensorly.decomposition import parafac
#import tensorly as tl
print("### Libraries loaded and locked", flush=True)

class advanced_AlexNet():
    """This class implements alexnet for classification of CIFAR 10 dataset
       along with CP decomposition applied to the specified layer of the pretrained alexnet.
       Implementation has been adapted from https://analyticsindiamag.com/implementing-alexnet-using-pytorch-as-a-transfer-learning-model-in-multi-class-classification/
    """
    def load_CIFAR_Data(self):
        """ This method loads the CIFAR 10 dataset and returns
        the training images, testing images and all the classes
        Input : 
        Output :
            trainloader : batch of training images
            testloader : batch of testing images
            classes : Number of classes
        """
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        ])
        train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)
        test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=2)
        classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
        print("### Dataset loaded and locked", flush=True)
        return trainloader, testloader, classes

    def get_Alexnet_Model(self):
        """ This method downloads the pretrained alexnet model 
            and changes the classification layer accordingly.
            Input :

            Output : 
                AlexNet_model : Model consisting of all the Alexnet parameters
        """
        AlexNet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
        AlexNet_model.classifier[4] = torch.nn.Linear(4096,1024)
        AlexNet_model.classifier[6] = torch.nn.Linear(1024,10)
        AlexNet_model.classifier = torch.nn.Sequential(*list(AlexNet_model.classifier) + [torch.nn.Softmax(10)])
        print("### Alexnet model loaded and locked", flush=True)
        print("Before changing", flush=True)
        print(AlexNet_model.eval(), flush=True)
        return AlexNet_model
    
    def compute_CPD_Alexnet(self, layer, AlexNet_model):
        """ This method replaces specified layer of alexnet with sequential 
        layer consisting of decomposed convolution weights
        Input : 
            layer : variable specifying the layer of the alesnet to be decomposed
            AlexNet_model : Alexnet model parameters
        Output :
            AlexNet_model : Alexnet model replaced with the decomposed parameters 
        """
        second_weight_tensor = AlexNet_model.features[3].weight.data
        max_iter = 100
        r = 64
        r_state = 0
        cp = CP_ALS()
        print("Computing the factors of the weight tensor", flush=True)
        start = time.time()
        A, lmbds = cp.compute_ALS(second_weight_tensor, max_iter, r)
        end = time.time()
        #x = parafac(tl.tensor(second_weight_tensor), rank=r, normalize_factors=False, init='random', random_state=r_state, n_iter_max= max_iter)
        #A = x[1]
        #K_t, K_s, K_y, K_x = torch.from_numpy(A[0]), torch.from_numpy(A[1]), torch.from_numpy(A[2]), torch.from_numpy(A[3])
        print("Factors calculated in ", end-start," seconds", flush=True)
        print("Factors shapes are: ", flush=True)
        K_t, K_s, K_y, K_x = A[0], A[1], A[2], A[3]

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
        print("After changing", flush=True)
        print(AlexNet_model.eval(), flush=True)
        return AlexNet_model
    
    def train_Alexnet(self):
        """ This method acts as meta method, becuase this method loads the dataset, 
            loads alexnet, computes the CP decomposition of specified layer and then trains the alexnet
            , finally then computes the classification accuracy.
            Input :

            Output :
        """
        trainloader, testloader, classes = self.load_CIFAR_Data()
        AlexNet_model = self.get_Alexnet_Model()
        layer = 3
        AlexNet_model = self.compute_CPD_Alexnet(layer, AlexNet_model)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device, flush=True)
        AlexNet_model.to(device)
        #criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.MSELoss()
        optimizer = optim.SGD(AlexNet_model.parameters(), lr=0.001, momentum=0.01)
        print("### Optimizer loaded and locked", flush=True)
        print("### Training started ", flush=True)
        for epoch in range(2):  
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                output = AlexNet_model(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:   
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        print('Finished Training of AlexNet', flush=True)
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
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        avg = 0
        for i in range(10):
            temp = (100 * class_correct[i] / class_total[i])
            avg = avg + temp
        avg = avg/10
        print('Average accuracy = ', avg)
        return 0






