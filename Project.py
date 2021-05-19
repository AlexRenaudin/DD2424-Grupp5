# Import
import numpy as np
import torch
import torch.nn as nn
import torchtext
from torchtext.datasets import PennTreebank

# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
n_epochs = 10
eta = 0.01
seq_len  = 25
momentum = 0.99
m = 100
variant = 'lstm' #rnn, lstm, 2-lstm, bi-lstm, gru
use_convnet = False
convnet_out_channels = 5
use_torchdata = False 

# Unique Characters
def unique_count(unique_text):
    unique = {}
    c_count = 0
    for line in unique_text:
        for word in line:
            for c in word:
                unique[c] = True
                c_count += 1
    return c_count, unique
if use_torchdata:
    train_unique_text, val_unique_text = PennTreebank(split=('train','valid'))
    train_count, unique = unique_count(train_unique_text)
    val_count, _ = unique_count(val_unique_text)
else:
    f = open('goblet_book.txt','r')
    unique_text = f.readlines()
    full_count, unique = unique_count(unique_text)

# Dictionaries 
int2char = {}
char2int = {}
K = 0
for c in unique:
    int2char[K] = c
    char2int[c] = K
    K += 1

# Data
try:
    train_tensor = torch.load('train_tensor.pt')
    val_tensor = torch.load('val_tensor.pt')
except FileNotFoundError:        
    def text2tensor(n_seq, text, seq_len, char2int):
        tensor = torch.zeros(seq_len, n_seq)
        i = 0
        for line in text:
            for word in line:
                    for c in word:
                        tensor[i%seq_len, i//seq_len] = char2int[c]
                        i += 1
        return tensor
    if use_torchdata:
        train_text, val_text = PennTreebank(split=('train', 'valid'))
        train_tensor = text2tensor((train_count//seq_len), train_text, seq_len, char2int)
        val_tensor = text2tensor((val_count//seq_len), val_text, seq_len, char2int)
    else:
        full_text = f.readlines()
        n_seq = (full_count//seq_len)
        full_tensor = text2tensor(n_seq, full_text, seq_len, char2int)
        val_size = n_seq//10
        train_tensor = full_tensor[:,0:n_seq-val_size]
        val_tensor = full_tensor[:,n_seq-val_size:n_seq]        
    torch.save(train_tensor, 'train_tensor.pt')   
    torch.save(val_tensor, 'val_tensor.pt')
        
# Network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, variant, use_convnet, out_channels):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.variant = variant
        self.use_convnet = use_convnet
        self.num_layers = 1
        self.num_directions = 1
        if variant == 'rnn':
            self.model = nn.RNN(input_size, hidden_size)   
        elif variant == 'lstm':
            self.model = nn.LSTM(input_size, hidden_size)
        elif variant == '2-lstm':
            self.num_layers = 2
            self.model = nn.LSTM(input_size, hidden_size, num_layers = self.num_layers)
        elif variant == 'bi-lstm':
            self.num_directions = 2
            self.model = nn.LSTM(input_size, hidden_size, bidirectional = True)
        elif variant == 'gru':
            self.model = nn.GRU(input_size, hidden_size)
        else:
            print('Unexpected variant')
            raise
        if use_convnet:       
            self.conv = nn.Conv1d(1, out_channels, kernel_size = 3, padding = 1)
            self.relu = nn.ReLU()
            self.flat = nn.Flatten()
            self.connected = nn.Linear(self.num_directions*out_channels*hidden_size, input_size)
        else:   
            self.linear = nn.Linear(self.num_directions*hidden_size, input_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_size).to(device)
        if self.variant == 'lstm' or self.variant == '2-lstm' or self.variant == 'bi-lstm':
            c0 = torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_size).to(device)
            x, _ = self.model(x, (h0, c0))           
        else:
            x, _ = self.model(x, h0)
        if self.use_convnet:
            x = self.conv(x)
            x = self.relu(x)
            x = self.flat(x)
            x = self.connected(x)    
        else:  
            x = self.linear(x)
            x = x[:,-1,:]
        return x

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

# Initialization 
net = Net(K, m, variant, use_convnet, convnet_out_channels).to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=eta)
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(n_epochs):
    for i in range(train_tensor.size(1)):
        X = train_tensor[:,i]
        target_Y = torch.cat((train_tensor[1:seq_len,i],train_tensor[0:1,i+1]),0).long()
        forward_Y = net(encode(X, K)[:,None,:])
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

# Validation
for i in range(val_tensor.size(1)):
    X = val_tensor[:,i]
    target_Y = torch.cat((val_tensor[1:seq_len,i],val_tensor[0:1,i+1]),0).long()
    forward_Y = net(encode(X, K)[:,None,:])
    loss = criterion(forward_Y, target_Y)
    loss.backward()
    if i/100 == i//100: 
        print(i)
        print('loss =', loss.item())
text = generate(1000, char2int, int2char, K)
print(text)
