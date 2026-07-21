
cuda = True if torch.cuda.is_available() else False


net = EVANet(num_class=40).cuda()

model = EG_DDPM(EEG_Size = 512)
generator = EGDDPM_Algorithm(
    model,
    timesteps= 30,  # number of steps
    c_l= 1          # 0: cosine     1: Linear
).cuda()


progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
for i, (eeg) in progress_bar:
    with torch.no_grad():
        eeg = eeg.to(device=device, dtype=torch.float)
		eeg = net(eeg)
        recon_image = generator.sample(batch_size=1, eeg=eeg)

