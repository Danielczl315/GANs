import copy

max_epoch = 50
binary = nn.BCELoss()
k = 5
D = Discriminator()
D.cuda()
G = Generator()
G.cuda()
g_loss = []
d_loss = []
d_optim = optim.RMSprop(D.parameters(), lr=0.0001)
g_optim = optim.RMSprop(G.parameters(), lr=0.0001)
for epoch in range(max_epoch):
    
    print('\n' + 'Epoch {}/{}'.format(epoch+1, max_epoch))
    print('-' * 20)
    start = time.time()
    g_error = 0.0
    d_error = 0.0
    for i, data in enumerate(data_loader):
        img, label = data
        # real
        img = img.to(device)
        label = label.to(device)
        # Train D
        d_optim.zero_grad()
        
        noises = Variable(torch.randn(100, 128)).to(device)
        fake_samples = G(noises).detach()
        
        # real, close to 1
        real_pred = D(img)
        real_loss = binary(real_pred.squeeze(), Variable(torch.ones(100)).to(device))
        
        # fake, close to 0
        fake_pred = D(fake_samples)
        fake_loss = binary(fake_pred.squeeze(), Variable(torch.zeros(100)).to(device))
                
        loss = real_loss + fake_loss
        loss.backward()
        d_error += loss
        d_optim.step()

        # make a copy of D into D unroll
        D_unroll = copy.deepcopy(D)
        d_unroll_optim = optim.RMSprop(D_unroll.parameters(), lr=0.0001)
        # D unroll for k steps
        for j, data_unroll in enumerate(data_loader):
            if j == k:
                break
            img_unroll, _ = data_unroll
            img_unroll = img_unroll.to(device)
            
            d_unroll_optim.zero_grad()
            noises = Variable(torch.randn(100, 128)).to(device)
            fake_samples = G(noises).detach()

            real_pred = D_unroll(img_unroll)
            real_loss = binary(real_pred.squeeze(), Variable(torch.ones(100)).to(device))

            fake_pred = D_unroll(fake_samples)
            fake_loss = binary(fake_pred.squeeze(), Variable(torch.zeros(100)).to(device))

            loss = real_loss + fake_loss
            loss.backward()
            d_unroll_optim.step() 

        # train G
        g_optim.zero_grad()
        
        noises = Variable(torch.randn(100, 128)).to(device)
        fake_samples = G(noises)

        pred = D_unroll(fake_samples)
        # as close as possible to 1
        loss = binary(pred.squeeze(), Variable(torch.ones(100)).to(device))
        loss.backward()
        g_error += loss
        g_optim.step()

    g_loss.append(float(g_error)/i)
    d_loss.append(float(g_error)/i)
    print('g_loss: {:.3} | d_loss: {:.3}\r'.format(float(g_error)/i, float(d_error)/i))
    print('Time: {}'.format(time.time()-start))