
import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

from PIL import Image

def load_checkpoint(checkpointpath, device):
    if device == 'cpu':
        map_location = 'cpu'
    else:
        map_location = 'cuda'
    checkpoint = torch.load(checkpointpath,map_location='cpu')
    model = checkpoint["model"]
    #model.to(device)
    #model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['model_class_to_idx']
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image_path)
    width, height  = img.size
    
    # resize
    ratio = float(width) / float(height)
    if ratio <=1:
        imsize = 256, int(256/ratio)
    else:
        imsize = int(256*ratio), 256
    img = img.resize(imsize)
    #crop the center
    crop_size = 224
    img = img.crop(((img.width - crop_size)//2,
                   (img.height - crop_size)//2,
                   (img.width + crop_size)//2,
                   (img.height + crop_size)//2))
    
    np_image = np.array(img, dtype = 'f')
    
    np_image = np_image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2,0,1))
    
    tensor_image = torch.from_numpy(np_image).float()
    
    return tensor_image

def predict(image_path, model, topk, device, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    #Implement the code to predict the class from an image file
    model.to(device)
    image = process_image(image_path)
    image.unsqueeze_(0)
    image = image.to(device)
    model.eval()
    #inverse map
    idx_to_class = {idx:mapclass for mapclass, idx in model.class_to_idx.items() }
    
    with torch.no_grad():
        ps = torch.exp(model(image))
        probs, top_classes = ps.topk(topk, dim=1)
        probs = probs.squeeze().tolist()
        classes = [idx_to_class.get(x) for x in top_classes.squeeze().tolist()]
    names = [cat_to_name[x] for x in classes]   
    
    [ print("flowers: {}, probabilities: {}".format(names[i],probs[i])) for i in range(topk)]
    return probs, classes

def main():
    """
    Example to run: python predict.py flowers/test/100/image_07897.jpg netcheckpoint.pth --top_k=5 --gpu
    
    """
    parser = argparse.ArgumentParser(description = "Load model and make prediction")
    parser.add_argument("image_directory", help = "path to image")   
    parser.add_argument("checkpoint_path")
    parser.add_argument("--category_names",default ="cat_to_name.json")
    parser.add_argument("--top_k", type=int, default=5, help="top k classes")
    parser.add_argument("--gpu", action="store_true", help="if GPU is available")
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu else "cpu")
    image_path = os.path.abspath(args.image_directory)
    checkpointpath = os.path.abspath(args.checkpoint_path) 
    topk = args.top_k
    category_names = args.category_names
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)    
    model= load_checkpoint(checkpointpath, device)
    probs, classes = predict(image_path, model, topk, device, cat_to_name)
      

    return None


if __name__ == "__main__":
    main()
    




