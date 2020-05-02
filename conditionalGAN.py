class ControlGAN():
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
        fake_labels = Variable(torch.randint(9,(self.batch_size,)).to(device))
        fake_oh = F.one_hot(fake_labels, num_classes=10)
        noises = torch.cat((noises, fake_oh.float()), axis=1)
        fake_samples = self.G(noises)

        pred = self.D(fake_samples, fake_labels)
        # as close as possible to 1
        loss = self.criterion(pred.squeeze(), Variable(torch.ones(self.batch_size)).to(device))
        loss.backward()
        self.g_optim.step()
        
        return loss
        
        
    def train_D(self, real_images, labels):
        """
        Get a batch of real images
        Get a batch of fake samples from G
        Optimize D to correctly classify the two batches
        """
        self.d_optim.zero_grad()
        
        noises = Variable(torch.randn(self.batch_size, 128)).to(device)
        fake_labels = Variable(torch.randint(9,(self.batch_size,)).to(device))
        fake_oh = F.one_hot(fake_labels, num_classes=10)
        noises = torch.cat((noises, fake_oh.float()), axis=1)

        fake_samples = self.G(noises).detach()
        
        # real, close to 1
        real_pred = self.D(real_images, labels)
        real_loss = self.criterion(real_pred.squeeze(), Variable(torch.ones(self.batch_size)).to(device))
        #real_loss.backward()
        
        # fake, close to 0
        fake_pred = self.D(fake_samples, labels)
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
        
        noises = Variable(torch.randn(self.batch_size, 128)).to(device)
        fake_labels = Variable(torch.randint(9,(self.batch_size,)).to(device))
        fake_oh = F.one_hot(fake_labels, num_classes=10)
        noises = torch.cat((noises, fake_oh.float()), axis=1)
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
                d_error += self.train_D(img, label)
                g_error += self.train_G()

            img = self.G(noises).cpu().detach()
            img = make_grid(img)
            self.images.append(img)
            self.g_losses.append(float(g_error)/i)
            self.d_losses.append(float(d_error)/i)
            print('g_loss: {:.3} | d_loss: {:.3}\r'.format(float(g_error)/i, float(d_error)/i))
            print('Time: {}'.format(time.time()-start))

cG = Generator()
cD = Discriminator()
cG.cuda()
cD.cuda()
criterion = nn.BCELoss()
c_g_optim = optim.RMSprop(cG.parameters(), lr=0.0001)
c_d_optim = optim.RMSprop(cD.parameters(), lr=0.0001)
cGan = ControlGAN(cG,cD,c_g_optim, c_d_optim, criterion)

cGan.train(data_loader, 100, BATCH_SIZE)

idx_to_class = dict(zip(fmnist.class_to_idx.values(), fmnist.class_to_idx.keys()))

for i in range(10):
  n = Variable(torch.randn(3, 128)).to(device)
  fake_labels = Variable((torch.ones(3)*i).to(device))
  fake_oh = F.one_hot(fake_labels.long(), num_classes=10)
  noises = torch.cat((n, fake_oh.float()), axis=1)
  plt.figure(figsize=(2.5,7.5))
  plt.title(idx_to_class[i])
  plt.imshow(transforms.ToPILImage(mode='L')(make_grid(cGan.G(noises)).cpu()[0]),cmap='gray')
  plt.savefig('{}.png'.format(i))
