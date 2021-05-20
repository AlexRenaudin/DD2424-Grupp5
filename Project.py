# Import
import numpy as np
import torch
import torch.nn as nn
import torchtext
from torchtext.datasets import PennTreebank

# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
n_epochs = 100
eta = 0.001
alpha = 0.9
seq_len  = 25
m = 100
variant = 'rnn' #rnn, lstm, 2-lstm, bi-lstm, gru
layerxdirection = 1
if variant == '2-lstm' or variant == 'bi-lstm':
    layerxdirection = 2
use_convnet = False
convnet_out_channels = 5
use_torchdata = False
#file_name = 'names.txt'
file_name = 'goblet_book.txt'
#file_name = 'Hermione.txt'
#file_name = 'random.txt'

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
    unique_text = open(file_name,'r').readlines()
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
        train_tensor = text2tensor((train_count//seq_len)+1, train_text, seq_len, char2int)
        val_tensor = text2tensor((val_count//seq_len)+1, val_text, seq_len, char2int)
    else:
        full_text = open(file_name,'r').readlines()
        n_seq = (full_count//seq_len)+1
        full_tensor = text2tensor(n_seq, full_text, seq_len, char2int)
        val_size = n_seq//10
        train_tensor = full_tensor[:,0:n_seq-val_size]
        val_tensor = full_tensor[:,n_seq-val_size:n_seq]        
    #torch.save(train_tensor, 'train_tensor.pt')   
    #torch.save(val_tensor, 'val_tensor.pt')

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
            self.convfc = nn.Linear(self.num_directions*out_channels*hidden_size, input_size)
        else:   
            self.fc = nn.Linear(self.num_directions*hidden_size, input_size)

    def forward(self, x, H, C):
        if self.variant == 'lstm' or self.variant == '2-lstm' or self.variant == 'bi-lstm':
            x, (H, C) = self.model(x, (H.detach(), C.detach()))           
        else:
            x, H = self.model(x, H.detach())
            x = torch.clamp(x, -1, 1)  
        if self.use_convnet:
            x = self.conv(x)
            x = self.relu(x)
            x = self.flat(x)
            x = self.convfc(x)    
        else:  
            x = self.fc(x)
            x = x[:,-1,:]
        return x, H, C
        

# One-hot encoding
def encode(tensor, K):
    n = tensor.size()[0]
    new_tensor = torch.zeros(n, K)
    for i in range(n):
        new_tensor[i,tensor[i].long()] = 1 
    return new_tensor

# Text generator
def generate(H, C, n_gen, char2int, int2char, K, net):
    X_gen = torch.tensor([char2int['a']])
    X_gen = encode(X_gen, K)
    text_gen = ''
    H_gen = H
    C_gen = C
    for i in range(n_gen):
        y_gen, H_gen, C_gen = net(X_gen[:,None,:], H_gen, C_gen)
        p_gen = y_gen[0].detach().numpy()
        p_gen =  np.exp(p_gen)/sum(np.exp(p_gen))
        sample = np.random.choice(a = range(K), p = p_gen)
        text_gen = text_gen + int2char[sample]
        X_gen = y_gen
    return text_gen

# Initialization 
net = Net(K, m, variant, use_convnet, convnet_out_channels).to(device)
optimizer = torch.optim.RMSprop(net.parameters(), lr=eta, alpha = alpha)
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(n_epochs):
    H = torch.zeros(layerxdirection, 1, m).to(device)
    C = torch.zeros(layerxdirection, 1, m).to(device)
    for i in range(train_tensor.size(1)-1):
        X = train_tensor[:,i]
        target_Y = torch.cat((train_tensor[1:seq_len,i],train_tensor[0:1,i+1]),0).long()
        forward_Y, H, C = net(encode(X, K)[:,None,:], H, C)
        loss = criterion(forward_Y, target_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i == 0:
            smooth_loss = loss.item()
        else:
            smooth_loss = 0.999* smooth_loss + 0.001*loss.item()            
        if i/100 == i//100: 
            print(i)
            print('smooth loss =', smooth_loss)
        if i/100 == i//100: 
            text = generate(H, C, 200, char2int, int2char, K, net)
            print(text)

# Validation
for i in range(val_tensor.size(1)-1):
    X = val_tensor[:,i]
    target_Y = torch.cat((val_tensor[1:seq_len,i],val_tensor[0:1,i+1]),0).long()
    forward_Y, H, C = net(encode(X, K)[:,None,:], H, C)
    loss = criterion(forward_Y, target_Y)
    loss.backward()
    if i/100 == i//100: 
        print(i)
        print('loss =', loss.item())
text = generate(H, C, 1000, char2int, int2char, K, net)
print(text)
