import torch     
import torch.nn as nn  #Contains neural network building blocks(lyers, loss functions)
from torchvision import models  #provides pre-built CNN architectures (ResNet, ConvNeXt, ecc.), optionally with pretrained weights on ImageNet.

#This function builds a CNN (ResNet18, ResNet50 and ConvNext-Tiny), optinally with pretrained ImageNet weights and replaces the final classification
#head with a regression head (1 neuron) so the model predicts the number of faces in an image.
def build_model(name='resnet18', pretrained=True):
    name = name.lower()   # "ResNet18" and "resnet18" are the same
    if name == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None) #pretrained=True->loads weights pretrained on ImageNet; pretrained=False -> initializes with random weights
        in_f = m.fc.in_features  #number of input features to the FC (fully connected(classification))layers
        m.fc = nn.Linear(in_f, 1) #the model outputs 1 value (the number of faces) instead of 1000 classes
        return m         
    if name == 'resnet50': #We use Resnet50 if we want a stronger model, at the cost of more computation.
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, 1)
        return m
    if name == 'convnext_tiny':   #A modern CNN inspired by Transformers but implemented as a convolutional network.
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        in_f = m.classifier[2].in_features   #The classification head is stored inside the last layer of ConvNeXt, m.classifier[2]. We replace it with a single output layer for face counting task.
        m.classifier[2] = nn.Linear(in_f, 1)
        return m   #Returns the model with the new regression head.
    raise ValueError(f'Model {name} non supportato')  #the function raises a ValueError if the user passes a model name that is not 'resnet18/50' or 'convnext_tiny'
