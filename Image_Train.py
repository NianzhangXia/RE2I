
cuda = True if torch.cuda.is_available() else False


batch_size = 64


model = EG_DDPM(EEG_Size = 512)
generator = EGDDPM_Algorithm(
    model,
    timesteps= 30,  # number of steps
    c_l= 1          # 0: cosine     1: Linear
).cuda()

optimizer = optim.Adam(generator.parameters(), lr=1e-4)
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

num_c = 0
for epoch in range(0, 500):
    run_loss = 0
    generator.train()
    for j,(eeg, img) in enumerate(dataloader):
        batch_size = eeg.shape[0]

        eeg = Variable(eeg.type(FloatTensor))
        real_imgs = Variable(img.type(FloatTensor))

        loss, fake = generator(x=real_imgs, eeg = eeg)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        run_loss += loss.item()

    print("epoch:{}, loss:{:.4f}".format(epoch, run_loss))


