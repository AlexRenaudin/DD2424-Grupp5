# Import
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Unique Characters
unique_text = PennTreebank(split='train')
unique = Counter()
for line in unique_text:
    for word in line:
        for c in word:
            unique.update(c)

# Parameters
n_epochs = 1
eta = 0.0001
seq_len  = 25
momentum = 0.99
m = 100
K = len(unique)

# Dictionaries 
int2char = {}
char2int = {}
k = 0
for c in unique:
    int2char[k] = c
    char2int[c] = k
    k += 1
'''
# One-Hot Encoding
def one_hot_encode(seq, K):
    one_hot = np.zeros((K, np.multiply(*seq.shape)), dtype=np.float32)
    one_hot[-1-seq.flatten(), np.arange(one_hot.shape[1])] = 1.
    return one_hot
'''
# Data
try:
    train_tensor = torch.load('train_tensor.pt')
    val_tensor = torch.load('val_tensor.pt')
    test_tensor = torch.load('test_tensor.pt')
except FileNotFoundError:
    train_text, val_text, test_text = PennTreebank()
    def text2tensor(length, text, seq_len, K, char2int):
        tensor = torch.zeros(seq_len, (length//seq_len)+1, K)
        i = 0
        for line in text:
            for word in line:
                for c in word:
                    tensor[i%seq_len, i//seq_len, char2int[c]] = 1
                    i += 1
        return tensor
    train_tensor = text2tensor(5101618, train_text, seq_len, K, char2int)
    torch.save(train_tensor, 'train_tensor.pt')
    val_tensor = text2tensor(399782, val_text, seq_len, K, char2int)
    torch.save(val_tensor, 'val_tensor.pt')
    test_tensor = text2tensor(449945, test_text, seq_len, K, char2int)
    torch.save(test_tensor, 'test_tensor.pt')

# Network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size)

    def forward(self, x):
        h0 = torch.zeros(1, self.input_size, self.hidden_size).to(device)
        xnext, _ = self.rnn(x, h0)
        xnext = xnext[:, -1, :]
        return xnext

# Initialization 
net = Net(seq_len, m).to(device)
optimizer = optim.SGD(net.parameters(), lr=eta, momentum = momentum)
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(n_epochs):
    for i in range(train_tensor.size(1)):
        X = train_tensor[:,i,:]
        Y = torch.cat((train_tensor[1:seq_len,i,:],train_tensor[seq_len-1:seq_len,i+1,:]),0)
        loss = criterion(net(X), Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(i)
