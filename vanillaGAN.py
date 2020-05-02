import torch
from torch.utils.data import *
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import torch.optim as optim
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid

BATCH_SIZE = 100

fmnist = torchvision.datasets.FashionMNIST(root="./", train=True,
transform = transforms.Compose([
        transforms.ToTensor()                                 
    ]), download=True)
data_loader = torch.utils.data.DataLoader(dataset=fmnist,
batch_size=BATCH_SIZE, shuffle=True)

for data in data_loader:
    img, label = data
    print(label)
    break

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.n_features = 128
        self.n_out = 28*28
        self.fc0 = nn.Sequential(
                    nn.Linear(self.n_features, 256),
                    nn.ReLU()
                    )
        self.fc1 = nn.Sequential(
                    nn.Linear(256, 512),
                    nn.ReLU()
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(512, 1024),
                    nn.ReLU()
                    )
        self.fc3 = nn.Sequential(
                    nn.Linear(1024, self.n_out),
                    nn.Tanh()
                    )
    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view(-1, 1, 28, 28)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_in = 28*28
        self.n_out = 1
        self.fc0 = nn.Sequential(
                    nn.Linear(self.n_in, 1024),
                    nn.ReLU()
                    )
        self.fc1 = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.ReLU()
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU()
                    )
        self.fc3 = nn.Sequential(
                    nn.Linear(256, self.n_out),
                    nn.Sigmoid()
                    )
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class VanillaGAN():
    def __init__(self, G, D, g_optim, d_optim, criterion):
        self.G = G
        self.D = D
        self.g_optim = g_optim
        self.d_optim = d_optim
        self.criterion = criterion
        self.images = []
        self.g_losses = []
        self.d_losses = []
        
    def train_G(self):
        """  
        Sample a batch of random noise
        Generate fake samples using the noise
        Feed fake samples to D and get prediction scores
        Optimize G to get the scores close to 1 (means real samples)
        """     
        self.g_optim.zero_grad()
        
        noises = Variable(torch.randn(self.batch_size, 128)).to(device)
        fake_samples = self.G(noises)

        pred = self.D(fake_samples)
        # as close as possible to 1
        loss = self.criterion(pred.squeeze(), Variable(torch.ones(self.batch_size)).to(device))
        loss.backward()
        self.g_optim.step()
        
        return loss
        
        
    def train_D(self, real_images):
        """
        Get a batch of real images
        Get a batch of fake samples from G
        Optimize D to correctly classify the two batches
        """
        self.d_optim.zero_grad()
        
        noises = Variable(torch.randn(self.batch_size, 128)).to(device)
        fake_samples = self.G(noises).detach()
        
        # real, close to 1
        real_pred = self.D(real_images)
        real_loss = self.criterion(real_pred.squeeze(), Variable(torch.ones(self.batch_size)).to(device))
        #real_loss.backward()
        
        # fake, close to 0
        fake_pred = self.D(fake_samples)
        fake_loss = self.criterion(fake_pred.squeeze(), Variable(torch.zeros(self.batch_size)).to(device))
        #fake_loss.backward()
                
        loss = real_loss + fake_loss
        loss.backward()
        self.d_optim.step()

        return loss
    
    def train(self, data_loader, num_epochs, batch_size):   
        self.batch_size = batch_size
        
        self.G.train()
        self.D.train()
        
        noise = Variable(torch.randn(self.batch_size, 128)).to(device)
        
        for epoch in range(num_epochs):
            start = time.time()
            print('\n' + 'Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 20)
            g_error = 0.0
            d_error = 0.0
            for i, data in enumerate(data_loader):
                img, label = data
                img = img.to(device)
                label = label.to(device)
                d_error += self.train_D(img)
                g_error += self.train_G()

            img = self.G(noise).cpu().detach()
            img = make_grid(img)
            self.images.append(img)
            self.g_losses.append(float(g_error)/i)
            self.d_losses.append(float(d_error)/i)
            print('g_loss: {:.3} | d_loss: {:.3}\r'.format(float(g_error)/i, float(d_error)/i))
            print('Time: {}'.format(time.time()-start))

myG = Generator()
myD = Discriminator()
myG.cuda()
myD.cuda()
criterion = nn.BCELoss()
g_optim = optim.Adam(myG.parameters(), lr=0.0001)
d_optim = optim.Adam(myD.parameters(), lr=0.0001)

myGAN = VanillaGAN(myG, myD, g_optim, d_optim, criterion)

myGAN.train(data_loader, 200, BATCH_SIZE)