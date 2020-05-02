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

class WDiscriminator(nn.Module):
    def __init__(self):
        super(WDiscriminator, self).__init__()
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
                    )
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class WGAN():
    def __init__(self, G, D, g_optim, d_optim, c):
        self.G = G
        self.D = D
        self.g_optim = g_optim
        self.d_optim = d_optim
        self.c = c
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
        fake_pred = self.D(fake_samples)
        # Expectation D(G(z))
        loss = -torch.mean(fake_pred)
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
        real_loss = torch.mean(real_pred)
        
        # fake, close to 0
        fake_pred = self.D(fake_samples)
        fake_loss = torch.mean(fake_pred)
                
        loss = -real_loss + fake_loss
        loss.backward()
        self.d_optim.step()

        # clip
        for p in self.D.parameters():
            p.data.clamp_(-self.c, self.c)

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
