import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import Net

# download CIFAR10 dataset, if the folder does not exist, it will be created
trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True, # if it is already downloaded, it is not downloaded again
                                        transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=4,
                                          shuffle=True, 
                                          num_workers=2)

if __name__ == '__main__':
    net=Net()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        runing_loss = 0.0
        for i, data in enumerate(trainloader,0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            runing_loss += loss.item()
            if i % 2000 == 1999:
                loss = runing_loss/2000
                print(f'epoch={epoch+1}, batch={i+ 1:5}, loss={loss:.2f}')
                runing_loss = 0.0
    print('Finished Training')