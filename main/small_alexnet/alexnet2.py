from alexnet_model import AlexNet
from collections import OrderedDict
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import matplotlib 
matplotlib.use('Agg')
import pylab as plt

class advancedAlexNet():
    """
    This class implements training of AlexNet for classification task
    on CIFAR 10 dataset.
    """
    def __init__(self):
        self.AlexNet_model = AlexNet()
    
    def load_CIFAR_10_Dataset(self):
        """ This method loads the CIFAR 10 dataset and returns
        the training images, testing images and all the classes
        Input : 
        Output :
            trainloader : batch of training images
            testloader : batch of testing images
            classes : Number of classes
        """
        transform = transforms.Compose([ transforms.Resize(64),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], 
                                                            std=[x/255.0 for x in [63.0, 62.1, 66.7]]),])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        torch.manual_seed(43)
        val_size = 5000
        train_size = len(dataset) - val_size
        train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
        test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)
        valloader = torch.utils.data.DataLoader(val_data, batch_size=4, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=2)
        classes = dataset.classes
        return trainloader, valloader, testloader, classes

    
    def initialize_weights(self, model):
        if type(model) in [nn.Linear, nn.Conv2d]:
            torch.nn.init.xavier_normal_(model.weight, gain=1.414213)
            model.bias.data.fill_(0.01)
        
    def validate_set(self, loader, epoch, criterion, device):
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for data in loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = self.AlexNet_model(images)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, total_loss, accuracy))
        return total_loss

    def fit_N_Test(self, epochs):
        trainloader, valloader, testloader, classes = self.load_CIFAR_10_Dataset()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.AlexNet_model.parameters(), lr=0.001, momentum=0.9)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: ",device, flush=True)
        self.AlexNet_model.apply(self.initialize_weights)
        self.AlexNet_model.to(device)
        val_loss = []
        train_loss = []
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            total_running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                output = self.AlexNet_model(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                total_running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                            (epoch + 1, i + 1, running_loss / 2000), flush=True)
                    running_loss = 0.0
            if epoch%5 == 0:
                torch.save(self.AlexNet_model.state_dict(), "./model_weights")
            print("Epoch [{}], train_loss: {:.4f}".format(epoch, total_running_loss), flush=True)
            train_loss.append(total_running_loss)
            valid_loss = self.validate_set(valloader, epoch, criterion, device) 
            val_loss.append(valid_loss) 
            print()
        plt.figure(figsize=(10, 10))
        plt.plot(train_loss, c="b", label="Training loss")
        plt.plot(val_loss, c="r", label="Validation loss")
        plt.legend()
        plt.grid()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig("Accuracies.png")
        self.test(testloader, criterion, device)

    def test(self, testloader, criterion, device):
        #Testing Accuracy
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = self.AlexNet_model(images)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print()
        print("Test loss: ", total_loss, flush=True)
        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total), flush=True)


adv_net = advancedAlexNet()
adv_net.fit_N_Test(15)