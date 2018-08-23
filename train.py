#all imports here
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
import json
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        help='Images Folder')
    parser.add_argument('--gpu', dest='gpu',
                        action='store_true', help='Train with GPU?')
    parser.set_defaults(gpu=False)
    
    architectures = {'densenet121','vgg16'}
    parser.add_argument('--save_dir', type=str,
                        help='Directory to save checkpoints')
    
    parser.add_argument('--arch', dest='arch', default='vgg16', action='store',
                        choices=architectures,
                        help='Architecture to use')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Model learning rate')
    parser.add_argument('--hidden_layers', type=int, default=512,
                        help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of epochs to train')
    
    return parser.parse_args()

def do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, device='cpu'):
    model.train()
    epochs = epochs
    print_every = print_every

    # change to cuda
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for count, (inputs, labels) in enumerate(trainloader):

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if count % print_every == 0:
                validation_loss, accuracy = check_accuracy_on_test(model, criterion, dataloaders['valid'],device)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.4f}".format(validation_loss),
                      "Accuracy: {:.4f}".format(accuracy))
                
                running_loss = 0
def check_accuracy_on_test(model, criterion, testloader, device):  
    model.to(device)
    model.eval()
    accuracy = 0
    test_loss = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward(images)
            test_loss += criterion(outputs, labels).item()
            ps = torch.exp(outputs).data 
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()
            
    return  test_loss/len(testloader), accuracy/len(testloader)

args = get_args()
data_transforms = {'train' :  transforms.Compose([transforms.RandomRotation(30),
                                                     transforms.RandomResizedCrop(224),
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                                          [0.229, 0.224, 0.225])]),
                      'test':    transforms.Compose([transforms.Resize(256),
                                                     transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                                          [0.229, 0.224, 0.225])])}
train_dir = args.data_dir + '/train/'
valid_dir = args.data_dir + '/valid/'
test_dir = args.data_dir + '/test/'
# TODO: Load the datasets with ImageFolder
image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                  'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['test']),
                  'test' : datasets.ImageFolder(test_dir,  transform=data_transforms['test'])}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
               'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
               'test'  : torch.utils.data.DataLoader(image_datasets['test'],  batch_size=64, shuffle=True)}
print("!@#",dataloaders)
if (torch.cuda.is_available() and args.gpu):
    device = 'cuda:0'
    print('GPU is traninig the model. It will be fast')
else:
    device = 'cpu'
    print('CPU is trianing the model. It might be slower')
#priting checkpoint save directory
if(args.save_dir):
    print("Checkpoint save directory: {}".format(args.save_dir))
#Printing Model Hyperparameters
print("Hyper parameters: ")
print("Architecture: {}".format(args.arch))
print("Learning rate: {}".format(args.learning_rate))
print("Hidden units: {}".format(args.hidden_layers))
print("Epochs: {}".format(args.epochs))
if (args.arch == 'vgg16'):
    print('Building model from vgg16 architecture')
    model = models.vgg16(pretrained=True)
    input_size = model.classifier[0].in_features
elif (args.arch == 'densenet121'):
    print('Building model from densnet121 architecture')
    model = models.densenet121(pretrained=True)
    input_size = model.classifier.in_features

for parameters in model.parameters():
    parameters.requires_grad = False
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size,args.hidden_layers)),
                                        ('ReLu', nn.ReLU()),
                                        ('fc2',nn.Linear(args.hidden_layers,102)),
                                        ('output',nn.LogSoftmax(dim=1))]))
model.classifier = classifier 
#Defining the loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
do_deep_learning(model, dataloaders['train'], args.epochs, 40, criterion, optimizer, 'cuda:0')

#testing the network with test data
test_loss, test_accuracy = check_accuracy_on_test(model,criterion, dataloaders['test'], 'cuda:0')
print("Loss: {:.4f}".format(test_loss),
      "Accuracy: {:.4f}".format(test_accuracy))

#Save class_to_idx from the dataset into the model
model.class_to_idx = image_datasets['train'].class_to_idx

# TODO: Save the checkpoint 
checkpoint = {'arch': args.arch,
              'state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'epochs': args.epochs,
              'print_every': 40,
              'classifier': model.classifier,
              'class_to_idx': model.class_to_idx,
              'learning_rate': args.learning_rate
             }
if args.save_dir:
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_path = args.save_dir + '/' + args.arch + '_checkpoint.pth'
else:
    save_path = args.arch + '_checkpoint.pth'
print("Saving checkpoint to {}".format(save_path))
torch.save(checkpoint, save_path)