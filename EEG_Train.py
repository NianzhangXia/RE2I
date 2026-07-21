
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    accu_num = 0
    sample_num = 0
    for i,(eeg, labels) in enumerate(data):
        sample_num += eeg.shape[0]
        eeg, labels = Variable(eeg.type(torch.cuda.FloatTensor)), Variable(labels.type(torch.cuda.LongTensor))

        _, output = model(eeg)
        pred = torch.max(output, dim=1)[1]

        accu_num += torch.eq(pred, labels).sum()

    return accu_num/sample_num


net = EVANet(num_class=40).cuda()

optimizer = optim.Adam(net.parameters(), lr = 0.001,betas=(0.5, 0.999))

max_val = 0
max_test = 0
test_line = 0
for epoch in range(200):
    running_loss = 0.0
    accu_num = 0
    sample_num = 0

    net.train()
    for i,(eeg, label) in enumerate(dataloader):
        sample_num += eeg.shape[0]
        eeg, label = Variable(eeg.type(torch.cuda.FloatTensor)) , Variable(label.type(torch.cuda.LongTensor))

        optimizer.zero_grad()
        _, outputs = net(eeg)

        loss = criterion(outputs, label)
        running_loss += loss.item()

        pred_classes  = torch.max(outputs, dim=1)[1]
        accu_num += torch.eq(pred_classes, label).sum()
        loss.backward()
        optimizer.step()

    trainacc = accu_num / sample_num
    valacc = evaluate(net, val_data)
    test_acc = evaluate(net, test_data)

    if valacc>max_val:
        max_val = valacc
        val_line = epoch
        max_test = test_acc
        test_line = epoch

    print("epoch:{};Training_L:{:.4f}; train _ {:.4f}; Val - {:.4f}; Test - {:.4f}; max_val's test acc':{:.4f};test_line:{}".format(
        epoch,running_loss,trainacc,valacc,test_acc,max_test,test_line))

