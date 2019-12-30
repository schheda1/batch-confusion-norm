
import numpy as np
import matplotlib.pyplot as plt

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


import pandas as pd
# from PIL import Image

from torchvision import models
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader

# from torchsummary import summary

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

class Cubs_dataset(Dataset):
  
  base_dir = 'CUB_200_2011/images'  

  def __init__(self, root, train=True, transform=None, loader=default_loader):
    self.root = os.path.expanduser(root)
    self.transform = transform
    self.train = train
    self.loader = default_loader
    self.__load_metadata__()
    # print('init')

  
  def __load_metadata__(self):
    images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ', names=['image_id', 'image_name'])
    labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'), sep=' ', names=['image_id', 'class_id'])
    train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'), sep=' ', names=['image_id', 'is_training_image'])
    class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'), sep=' ', names=['class_id', 'class_name'])
    
    data = images.merge(labels, on='image_id')
    self.data = data.merge(train_test_split, on='image_id')


    if self.train:
      self.data = self.data[self.data.is_training_image == 1]
    else:
      self.data = self.data[self.data.is_training_image == 0]


  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, index):
    sample = self.data.iloc[index]
    path = os.path.join(self.root, self.base_dir, sample.image_name)
    image = self.loader(path)
    #print(type(image))
    class_id = sample.class_id - 1

    #print(self.transform)
    if self.transform:
     image = self.transform(image)
    
    return image, class_id

transforms_train = transforms.Compose([transforms.Resize((600, 600)),
                                       transforms.RandomCrop(size=(448,448)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()
                                         ])
transforms_test = transforms.Compose([transforms.Resize((448, 448)),
                                         transforms.ToTensor()
                                        ])

cubs_dataset_train = Cubs_dataset(root='datasets',transform=transforms_train) # train #can use loader=pil_image
cubs_dataset_test = Cubs_dataset(root='datasets',train=False, transform=transforms_test)
print(len(cubs_dataset_train))
print(len(cubs_dataset_test))
print('loaded train and test sets')

validation_split = .03334
random_seed= 46
dataset_size = len(cubs_dataset_train)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))


print('training about to start')
base_lr = 8e-3
resnet = resnet.float().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.parameters(), lr = base_lr, momentum = 0.9)

num_epochs = 30
num_classes = 200
file = open("bcn_log.txt", "a")

l_ambda = 10
batch_size = 16
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10000)

print('training about to start')

for epoch in range(num_epochs):

  np.random.seed(random_seed+epoch)
  np.random.shuffle(indices)
  train_indices, val_indices = indices[split:], indices[:split]
  train_sampler = SubsetRandomSampler(train_indices)
  valid_sampler = SubsetRandomSampler(val_indices)
  
  D1 = DataLoader(cubs_dataset_train, batch_size=batch_size, sampler=train_sampler)
  total_step = len(D1)
  epoch_bcn = 0
  t_loss = 0
  for index, (images, labels) in enumerate(D1):

    # compute Cross Entropy Loss
    images = images.to(device)
    labels = labels.to(device)
    outputs = resnet(images)
    loss_ce = criterion(outputs, labels)

    # compute BCN loss as the sum of singular values of (softmax output*transpose(softmax output))  
    batch_p = F.softmax(outputs, dim=1)
    batch_p_t = batch_p.t()
    new_tensor = torch.mm(batch_p, batch_p_t)
    u, s, v = torch.svd(new_tensor, compute_uv=True)
    loss_bcn = sum(s)
    epoch_bcn += loss_bcn

    # total loss
    loss_batch = loss_ce + l_ambda * loss_bcn
    
    # training loss per epoch
    t_loss += loss_batch
    
    #nullify gradients and backpropagate computed loss
    optimizer.zero_grad()
    loss_batch.backward()
    optimizer.step()

    if (index + 1) % 50 == 0:
        print("Epoch: ", (epoch + 1), " Step: ", (index + 1), " /", total_step, " Loss: ", loss_batch.item())

  # average loss per sample  
  epoch_loss = t_loss.item()/total_step
  scheduler.step()
  #perform validation testing
  with torch.no_grad():
      correct = 0
      total = 0
      v_loss = 0
      r_loss = 0
      D3 = DataLoader(cubs_dataset_train, batch_size=batch_size,
                       sampler=valid_sampler)
      for ind, (images, labels) in enumerate(D3):
          images = images.to(device)
          labels = labels.to(device)
          outputs = resnet(images)
          v_loss += criterion(outputs, labels)
          r_loss += v_loss.item()
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
    
      acc = 100*correct/total
      #accuracy_epoch.append(acc)
      # string2 = "Epoch: " +str(epoch)+ " Accuracy: " + str(acc) +"\n"
      string3 = "Epoch: " + str(epoch)+" Training Loss: "+str(epoch_loss)+ " Validation Loss: "+ str(r_loss/len(D3)) + " BCN Confusion: " +str(epoch_bcn.item()) + "\n"
      file.write(string3)
      print('Accuracy: ', acc, " Validation Loss: ", r_loss/len(D3), " Training Loss: ", epoch_loss)

#mod_name = "model_bcn/model_7.pth"
#torch.save(resnet, mod_name)
print('training done')

#resnet = torch.load('model_bcn/model_7.pth')
#file=open("bcn_log.txt","a")
print('testing model...')
resnet.eval()  # eval mode 
with torch.no_grad():
    correct = 0
    total = 0
    D3 = DataLoader(cubs_dataset_test, batch_size=60, shuffle=False)
    for ind, (images, labels) in enumerate(D3):
        images = images.to(device)
        labels = labels.to(device)
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    final = 100*correct/total
    print('Test Accuracy of the model: {} %'.format(100*correct / total))
    string_final = "Test accuracy of the model: "+ str(final)+"\n"
    file.write(string_final)

file.close()

