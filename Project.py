# Import
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
n_epochs = 10
n_batch = 100
eta = 0.0001
T = 25 #seq_length 
momentum = 0.99
#???

# Data

#transform = ???, 

trainset = torchvision.datasets.???(root='./data', train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=n_batch, shuffle=True)

testset = torchvision.datasets.???(root='./data', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=n_batch, shuffle=False)

#classes = ???

# Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(80, 100)

    def forward(self, x):
        h0 = torch.zeros(1, 1, 100).to(device) 
        c0 = torch.zeros(1, 1, 100).to(device) 
        x = self.lstm(x, (h0, c0))  
        return x

# Training
net = Net().to(device)
optimizer = optim.SGD(net.parameters(), lr=eta, momentum = momentum)
criterion = nn.CrossEntropyLoss()
for epoch in range(n_epochs):
    for i, data in enumerate(train_loader):
        inputs, labels = data.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Generate Text

