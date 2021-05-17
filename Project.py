# Import
import numpy as np
import torch
import torch.nn as nn
import torchtext
from torchtext.datasets import PennTreebank

# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Unique Characters
unique_text = PennTreebank(split='train')
unique = {}
for line in unique_text:
    for word in line:
        for c in word:
            unique[c] = True

# Dictionaries 
int2char = {}
char2int = {}
k = 0
for c in unique:
    int2char[k] = c
    char2int[c] = k
    k += 1

# Parameters
n_epochs = 1
eta = 0.01
seq_len  = 25
momentum = 0.99
m = 100
K = len(unique)

# Data
try:
    train_tensor = torch.load('train_tensor.pt')
    val_tensor = torch.load('val_tensor.pt')
    test_tensor = torch.load('test_tensor.pt')
except FileNotFoundError:
    train_text, val_text, test_text = PennTreebank()
    def text2tensor(length, text, seq_len, char2int):
        n_seq = (length//seq_len)+1
        tensor = torch.zeros(seq_len, n_seq)
        i = 0
        for line in text:
            for word in line:
                for c in word:
                    tensor[i%seq_len, i//seq_len] = char2int[c]
                    i += 1
        return tensor
    train_tensor = text2tensor(5101618, train_text, seq_len, char2int)
    torch.save(train_tensor, 'train_tensor.pt')
    val_tensor = text2tensor(399782, val_text, seq_len, char2int)
    torch.save(val_tensor, 'val_tensor.pt')
    test_tensor = text2tensor(449945, test_text, seq_len, char2int)
    torch.save(test_tensor, 'test_tensor.pt')

# Network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        h0 = torch.zeros(1, 1, self.hidden_size).to(device) #1 layer and 1 batch
        c0 = torch.zeros(1, 1, self.hidden_size).to(device)
        x, _ = self.lstm(x, (h0, c0)) # x -> h
        x = self.linear(x) # h -> y
        x = x[:,-1,:] #Reduce from 3 to 2 dimensions
        return x

# Initialization 
net = Net(K, m).to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=eta)
criterion = nn.CrossEntropyLoss()

# One-hot encoding
def encode(tensor, K):
    n = tensor.size()[0]
    new_tensor = torch.zeros(n, K)
    for i in range(n):
        new_tensor[i,tensor[i].long()] = 1 
    return new_tensor

# Text generator
def generate(n, char2int, int2char, K):
    X = torch.tensor([char2int['.']])
    X = encode(X, K)
    text = '.'
    for i in range(n):
        y = net(X[:,None,:])
        p = y[0].detach().numpy()
        p =  np.exp(p)/sum(np.exp(p))
        sample = np.random.choice(a = range(K), p = p)
        text = text + int2char[sample]
        X = y
    return text

# Training
for epoch in range(n_epochs):
    for i in range(train_tensor.size(1)):
        X = train_tensor[:,i]
        target_Y = torch.cat((train_tensor[1:seq_len,i],train_tensor[0:1,i+1]),0).long()
        forward_Y = net(encode(X, K)[:,None,:]) #Raise from 2 to 3 dimensions (only one batch)
        loss = criterion(forward_Y, target_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i/100 == i//100: 
            print(i)
            print('loss =', loss.item())
        if i/100 == i//100: 
            text = generate(100, char2int, int2char, K)
            print(text)
