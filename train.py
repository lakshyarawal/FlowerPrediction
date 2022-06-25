import argparse
parser = argparse.ArgumentParser()
print("Hello World")
import time
import json
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from PIL import Image
parser.add_argument("learning_rate", help = "The rate at which the model will learn" , type = float)
parser.add_argument("data_dir", help = "The directory of dataset" )
parser.add_argument("model_architecture", help = "The architechture of model to be trained" )
parser.add_argument("epochs", help = "The epochs for which model is to be trained" ,type = int )
parser.add_argument("hidden_units", help = "The number of hidden units to be placed in the classifier" ,type = int )
parser.add_argument("gpu_choice", help = "The device preference for gpu/cpu" )
args = parser.parse_args()
hidden_units =  args.hidden_units
gpu_choice = args.gpu_choice
model_architecture = args.model_architecture
data_dir = args.data_dir
epochs = args.epochs
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
train_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
# TODO: Build and train your network
if(model_architecture=="vgg16"):
    model = models.vgg16(pretrained=True)
else:
    model = models.vgg19(pretrained=True)
model
for param in model.parameters():
    param.requires_grad = False
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088,4096)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(4096,hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(hidden_units,102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier
if(gpu_choice=="GPU"):
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
criterion = nn.NLLLoss()
 # Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
model.to(device)
for ii, (inputs, labels) in enumerate(trainloader):

# Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)
        start = time.time()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if ii==3:
            break
        
print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
epoch = epochs
print_every = 40
steps = 0

# change to cuda
model.to('cuda')

for e in range(epoch):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every))
            
            running_loss = 0

correct = 0
total = 0

with torch.no_grad():
    for data in validloader:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Implement a function for the validation pass
def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:

        #images.resize_(images.shape[0], 25088)
        images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy
epoch = epochs 
steps = 0
running_loss = 0
print_every = 40
for e in range(epoch):
    model.train()
    for images, labels in validloader:
        steps += 1
        
        # Flatten images into a 784 long vector
        #images.resize_(images.size()[0], 25088)
        images, labels = images.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(model, validloader, criterion)
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))
            
            running_loss = 0
            
            # Make sure training is back on
            model.train()
model.class_to_idx = train_data.class_to_idx
torch.save({'arch': model_architecture,
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx}, 
            'classifier.pth')