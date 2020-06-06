
#SON.ipynb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import time
import os
import copy
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import csv 
import cv2
import numpy
import glob
import torchvision.transforms.functional as F
import pandas as pd
from torch.autograd import Variable
from PIL import Image
import matplotlib.patches as patches

from cv2 import VideoWriter, VideoWriter_fourcc
from PIL import ImageDraw

from google.colab import drive
drive.mount('/content/drive',force_remount=True)
import scipy.io as sio

resnetmodel = models.resnext50_32x4d(pretrained=True)
resnetmodel.fc = nn.Linear(2048,3)

print(resnetmodel)

def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params

def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d



preprocess = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(64), 
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

])
class Custom_dataset(Dataset):

    def __init__(self, dataframe):
        self.dataframe = dataframe
      
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        t0path = self.dataframe.iloc[idx, 0]
       
        mat_path = self.dataframe.iloc[idx, 1]


        img = Image.open(t0path)

        # Crop the face loosely
        pt2d = get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        # k = 0.25 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
       
        poses = torch.FloatTensor([yaw, pitch, roll])
        
       
        
        
        t0tensor = preprocess(img).float()
        t0tensor = t0tensor.squeeze()
        t0Var = Variable(t0tensor)

     
        inputs = t0Var
        outputs = poses

        return (inputs, outputs)





csv_train = pd.read_csv("drive/My Drive/Colab Notebooks/train.csv")
val_train = pd.read_csv("drive/My Drive/Colab Notebooks/val.csv")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

valdataset = Custom_dataset(val_train)
valloader = torch.utils.data.DataLoader(valdataset, batch_size=64, shuffle=True, num_workers=4)

traindataset = Custom_dataset(csv_train)
trainloader = torch.utils.data.DataLoader(traindataset, batch_size=64, shuffle=True, num_workers=4)

datasetloaders = {'train': trainloader, 'val': valloader}

print(len(traindataset))
print(len(valdataset))
for i,y in trainloader:
  print(i.shape)
  print(y.shape)
  break
for i,y in valloader:
  print(i.shape)
  print(y.shape)
  break

print(len(traindataset))
print(len(valdataset))

def plot_graph(plotlist1,plotlist2,ylabel):
   #Plot accuracy graph 
    plt.xlabel("Training Epochs")
    plt.ylabel(ylabel)
    plt.plot(plotlist1, color="green")
    plt.plot(plotlist2, color="red")
    
    plt.gca().legend(('Train', 'Validation'))
    plt.show()
def regressionnetworktrain(mmodel,criterion,optimizer,dataloaders,epoch_number,device):
    mmodel.to(device)
    best_model_wts = copy.deepcopy(mmodel.state_dict())
    best_train_loss =  np.Inf
    train_loss_history =list()
    best_val_loss = np.Inf
    val_loss_history =list()
    
    for epoch in range(epoch_number):
        print('Epoch {}/{}'.format(epoch, epoch_number - 1))        
        # Each epoch has a training and validation phase
        for part in ['train', 'val']:
            current_loss = 0.0           
            if part == 'train':              
                mmodel.train()             
            else:
                mmodel.eval()  
            # For each phase in datasets are iterated
            for inputs,outputs in dataloaders[part]:

              inputs = inputs.to(device)
              outputs = outputs.to(device)
                 
              preds = mmodel(inputs)
              

              # zero the parameter gradients
              optimizer.zero_grad()
              # forward
              
              loss = criterion(preds, outputs)

              # Backpropagate and opitimize Training part
              if part == 'train':
                  loss.backward()
                  optimizer.step()
              # statistics
              current_loss += loss.item() * inputs.size(0)

            current_loss = current_loss /dataset_sizes[part]
            
            if part == 'val':      
                val_loss_history.append(current_loss)      
            else:                
                train_loss_history.append(current_loss)

            print('{} Loss: {:.4f} : '.format(
                part, current_loss))

            # deep copy the model
            if part == 'train' and current_loss < best_train_loss:
                  best_train_loss = current_loss
                
            if part == 'val' and current_loss  < best_val_loss:             
                best_val_loss = current_loss
                best_model_wts = copy.deepcopy(mmodel.state_dict())
           
        print()
    print('Best train Loss: {:.4f} : '.format(best_train_loss))  
    print('Best val Loss: {:.4f} : '.format(best_val_loss))

    # load best model weights
    mmodel.load_state_dict(best_model_wts)
    #Plot accuracy graph 
    
    plot_graph(train_loss_history,val_loss_history,"Loss")
  
    return mmodel

resnetmodel = models.resnet34(pretrained = True)
class myNetwork(nn.Module):
  def __init__(self):
      super(myNetwork, self).__init__()
      
      self.classifier = nn.Sequential(
            
            nn.Linear(512, 256),     
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 3),     
            
                 
        )

  def forward(self, x):   
    
    x = x.view(-1, 512)
    x = self.classifier(x)

    return x
resnetmodel.fc = myNetwork()

learning_rate = 0.0001
epoch =20

criterion = nn.L1Loss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_sizes = {'train': len(traindataset), 'val': len(valdataset)}


optimizer = optim.Adam(resnetmodel.parameters(), lr=learning_rate,weight_decay=0.005)


trained_model =  regressionnetworktrain(resnetmodel, criterion, optimizer,datasetloaders,epoch,device)



