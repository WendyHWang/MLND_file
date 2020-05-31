import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

from PIL import Image

import matplotlib.pyplot as plt
from utils import load_data

def reset_classifier(hidden_units):
    # updata fully-connected layers 
    classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(hidden_units, 102),
                               nn.LogSoftmax(dim=1))
    
    print('classifier: OK')
    return classifier
   
   
    
def train_model(netmodel, dataloaders,class_to_idx, abs_save_dir, lr, hidden_units, epochs, device):
    # Freeze parameters 
    for params in netmodel.parameters():
        params.requires_grad = False
    
    netmodel.classifier = reset_classifier(hidden_units)


    criterion = nn.NLLLoss()
    optimizer = optim.Adam(netmodel.classifier.parameters(), lr)

    netmodel.to(device)
    
    train_losses, valid_losses = [], []
    for e in range(epochs):
        train_loss = 0
        netmodel.train()
        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            log_ps = netmodel(images)
            loss = criterion(log_ps, labels)
            train_loss += loss
            loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                netmodel.eval()
                valid_loss = 0
                accuracy = 0
                for images, labels in dataloaders['valid']:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = netmodel(images)
                    valid_loss += criterion(log_ps, labels)
                    # accuracy
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor))

                train_losses.append(train_loss/len(dataloaders['train']))
                valid_losses.append(valid_loss/len(dataloaders['valid']))
                print("Epoch:{}/{}.. ".format(e+1, epochs),
                      "Train Loss:{:.3f}.. ".format(train_loss/len(dataloaders['train'])),
                      "Validation Loss:{:.3f}.. ".format(valid_loss/len(dataloaders['valid'])),
                      "Validation Accuracy:{:.3f}.. ".format(accuracy/len(dataloaders['valid'])*100))
                
    torch.save({"model": netmodel,
            "optimizer_state_dict":optimizer.state_dict(),
            "epoch": epochs,
            "model_class_to_idx": class_to_idx}, os.path.join(abs_save_dir,"netcheckpoint.pth"))

    return 



def main():

    parser = argparse.ArgumentParser(description = "Train a model and make prediction")
    parser.add_argument("data_directory", help = "path to data")
    parser.add_argument("--save_dir", default = "./",  help = "directory to save model")
    parser.add_argument("-lr","--learning_rate", type=float, default=0.005, help="learning rate for optimizer")
    parser.add_argument("--hidden_units", type=int, default = 512, help="input size of hidden layers" )
    parser.add_argument("--epochs", type=int, default = 20, help="epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--gpu", action="store_true", help="if GPU is available")
    parser.add_argument("--arch", default='vgg16', help = "pretrained model")
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu else "cpu")
    data_dir = os.path.abspath(args.data_directory)

    abs_save_dir = os.path.abspath(args.save_dir)

    lr = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    batch_size = args.batch_size

    if args.arch == 'vgg16':
        model = models.vgg16(pretrained = True)

    dataloaders, class_to_idx = load_data(data_dir, batch_size)
    train_model(model, dataloaders,class_to_idx, abs_save_dir, lr, hidden_units, epochs, device)
        
        


    return None


if __name__ == "__main__":
    main()
    








    