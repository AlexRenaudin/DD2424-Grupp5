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

# Vocabulary
vocab_iter = PennTreebank(split='train')
tokenizer = get_tokenizer('basic_english')
counter = Counter()
for line in vocab_iter:
    counter.update(tokenizer(line))
vocab = Vocab(counter)

# Data
train_iter, val_iter, test_iter = PennTreebank()
def data_process(raw_text_iter, vocab):
  data = [torch.tensor([vocab[token] for token in tokenizer(item)],
                       dtype=torch.float) for item in raw_text_iter]
  return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
train_data = data_process(train_iter, vocab)
val_data = data_process(val_iter, vocab)
test_data = data_process(test_iter, vocab)

# Parameters
n_epochs = 1
eta = 0.0001
seq_len  = 25
momentum = 0.99
m = 100
K = len(vocab)

# Network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        print(x.size())
        print(h0.size())
        print(c0.size())
        out, _ = self.lstm(x, (h0,c0))
        out = out[:, -1, :]
        return out

# Initialization 
net = Net(seq_len, m, 1).to(device)
optimizer = optim.SGD(net.parameters(), lr=eta, momentum = momentum)
criterion = nn.CrossEntropyLoss()
E = train_data.size(0) - seq_len

# Training
for epoch in range(n_epochs):
    e = 0
    while e <= E:
        X = train_data[e:e+seq_len]
        Y = train_data[e+1:e+seq_len+1]
        loss = criterion(net(X), Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        e = e + seq_length
        print(e)
