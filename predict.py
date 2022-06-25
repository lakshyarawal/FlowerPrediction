import argparse
parser = argparse.ArgumentParser()
print("Hello World")
from collections import OrderedDict
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
parser.add_argument("check_point", help = "The checkpoint where the model is stored")
parser.add_argument("file_path", help = "The path of the image file")
parser.add_argument("json_file", help = "The mapping file that maps classes to names")
parser.add_argument("top_k", help = "The number of classes that are required in result" , type =int)
parser.add_argument("gpu_choice", help = "The device preference for gpu/cpu" )
args = parser.parse_args()
gpu_choice = args.gpu_choice
top_k = args.top_k
json_file = args.json_file
model_checkpoint = args.check_point
def load_checkpoint(filepath):
    chpt = torch.load(filepath)
    
    if chpt['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True) 
        for param in model.parameters():
            param.requires_grad = False
    elif chpt['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Sorry base architecture note recognized") 
    model.class_to_idx = chpt['class_to_idx']
 # Create the classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088,4096)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(4096,2000)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(2000,102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        # Put the classifier on the pretrained network
    model.classifier = classifier
    model.load_state_dict(chpt['state_dict'])
    return model
model = load_checkpoint(model_checkpoint)
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
     # Define transformations for the image, should (note that imagenet models are trained with image size 224)
    transformation = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()
                                        ])
    # TODO: Process a PIL image for use in a PyTorch 
    image = Image.open(image_path)
    image = transformation(image).float()
    np_image = np.array(image)
    # Preprocess the image
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = transforms.ToPILImage()
    np_image = (np.transpose(np_image,(1,2,0))-mean)/std
    np_image = np.transpose(np_image,(2,0,1))
    return np_image
##Loading json for category Mapping

with open(json_file) as f:
    label_map = json.load(f)

image_path = 'flowers/test/1/image_06743.jpg'
img = process_image(image_path)
img.shape

def predict(image_path, model, topk=top_k):
    # Process image
    img = process_image(image_path)
    
    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze_(0)
    
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [label_map[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels, top_flowers
probs, labs, flowers = predict(image_path, model)
print(probs)
print(labs)
print(flowers)