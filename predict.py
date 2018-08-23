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
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser()
    
    #input file is mandatory
    parser.add_argument('input', type=str,
                        help='Test Image File')
    #checkpoint is mandatory
    parser.add_argument('checkpoint', type=str,
                        help='Saved model checkpoint')

    parser.add_argument('--top_k', type=int,
                        help='Return the top K most likely classes')
    parser.set_defaults(top_k=1)
    
    parser.add_argument('--category_names', type=str,
                        help='File of category names')

    parser.add_argument('--gpu', dest='gpu',
                        action='store_true', help='Use GPU')
    parser.set_defaults(gpu=False)

    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if (checkpoint['arch'] == 'densenet121'):
        loaded_model = models.densenet121(pretrained=True)
    elif (checkpoint['arch'] == 'vgg16'):
        loaded_model = models.vgg16(pretrained=True)
    loaded_model.classifier = checkpoint['classifier']
    loaded_epochs = checkpoint['epochs']
    loaded_print_every = checkpoint['print_every']
    loaded_model.load_state_dict(checkpoint['state_dict'])
    loaded_model.class_to_idx = checkpoint['class_to_idx']
    loaded_optimizer = optim.Adam(loaded_model.classifier.parameters(), checkpoint['learning_rate'])
    loaded_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Loaded the Model from checkpoint: {}\n"
           "Model has following properties\n"
           "Architecture: {}\n"
           "Learning Rate: {}\n"
           "Epochs: {}\n"
           "Validation Pass Frequency: {}\n"
           .format(filepath,checkpoint['arch'],checkpoint['learning_rate'],checkpoint['epochs'],checkpoint['print_every']))
    return loaded_model, loaded_optimizer, loaded_epochs, loaded_print_every

# TODO: Process a PIL image for use in a PyTorch model
def process_image(im):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    size = 256
    width, height = im.size
    #lower of width and height will be changed to size 224, changing the other one so as to keep the aspect ratio 
    if height > width:
        height = int(max(height * size / width, 1))
        width = int(size)
    else:
        width = int(max(width * size / height, 1))
        height = int(size)
    
    #resizing the image
    resized_image = im.resize((width, height))
    size = 224
    #setting up variables to crop the image
    x0 = (width - size) / 2
    y0 = (height - size) / 2
    x1 = x0 + size
    y1 = y0 + size
    
    # Image.crop(left, upper, right, lower)
    cropped_image = im.crop((x0, y0, x1, y1))
    
    #Nomalizing the color channels in 0 to 1 float range using numpy
    np_image = np.array(cropped_image) / 255. 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])    
    
    #Noramalizing the image as expected by the model 
    np_image_array = (np_image - mean) / std 
    
    #color channel needs to be first which is third in PIL image for PyTorch
    np_image_array = np_image.transpose((2, 0, 1)) 
    
    return np_image_array

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    PIL_image = Image.open(image_path)
    processed_image = process_image(PIL_image)
    image_tensor = torch.from_numpy(processed_image)
    model.to(device)
    with torch.no_grad():
        image_tensor = image_tensor.to(device).float()
        image_tensor.unsqueeze_(0)
        output = model.forward(image_tensor)
        ps = torch.exp(output)
    probs,indices = torch.topk(ps,topk)
    probs = np.array(probs[0])
    indices = np.array(indices[0])
    classes = [clas for clas,ind in model.class_to_idx.items() if ind in indices]
    return probs,classes
# TODO: Implement the code to predict the class from an image file
args = get_args()
if (torch.cuda.is_available() and args.gpu):
    device = 'cuda:0'
    print('GPU is predicting. It will be fast')
else:
    device = 'cpu'
    print('CPU is predicting. It could be slower')

#printing messgaes 
print("Input file: {}".format(args.input))
print("Checkpoint file: {}".format(args.checkpoint))
if args.top_k:
    print("Top {} most likely classes will be returned".format(args.top_k))
if args.category_names:
    print("Category names file is given: {}".format(args.category_names))
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    print("Provided Category Names Loaded")
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    print("Default Category Names Loaded")
#Loading back the model from saved checkpoint
print("Loading model...")
new_model,new_optimizer,new_epochs,new_print_every = load_checkpoint(args.checkpoint)
probabilities, classes = predict(args.input, new_model,device,args.top_k)
for a,b in zip(classes,probabilities):
    print("Flower is {:.9s} with probability {:.4f}".format(cat_to_name[a],b))