#################### Final Code ##################### 
# Import
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchtext
import torchvision.transforms as transforms
import process_data
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
n_epochs = 1
n_batch = 100
eta = 0.0001
T = 25 #seq_length 
momentum = 0.99
#???

# Data

#transform = ???, 

#trainset = torchtext.datasets.???(root='./data', train=True, transform=transform, download=True)
#trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=n_batch, shuffle=True)

#testset = torchtext.datasets.???(root='./data', train=False, transform=transform, download=True)
#testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=n_batch, shuffle=False)

#trainset = torchtext.datasets.PennTreebank(root='./data', split=('train', 'valid', 'test'))
#trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=n_batch, shuffle=True)

#testset = torchtext.datasets.PennTreebank(root='./data', train=False, transform=transform, download=True)
#testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=n_batch, shuffle=False)

trainloader = process_data.TBDataset().get_data()



#classes = ???

train_iter = PennTreebank(split='train')
tokenizer = get_tokenizer('basic_english')
counter = Counter()
for line in train_iter:
    counter.update(tokenizer(line))
vocab = Vocab(counter)

def data_process(raw_text_iter):
  data = [torch.tensor([vocab[token] for token in tokenizer(item)],
                       dtype=torch.float) for item in raw_text_iter]
  return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_iter, val_iter, test_iter = PennTreebank()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 9
eval_batch_size = 10
train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

bptt = 3
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


# Network
class Net(nn.Module):
    #Changed according to https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(3, 3)

    def forward(self, x):
        h0 = torch.zeros(1, 3, 3).to(device) 
        c0 = torch.zeros(1, 3, 3).to(device) 
        x = self.lstm(x, (h0, c0))
        return x

# Training

net = Net().to(device)
optimizer = optim.SGD(net.parameters(), lr=eta, momentum = momentum)
criterion = nn.CrossEntropyLoss()
for epoch in range(n_epochs):
    for i, data in enumerate(range(0, train_data.size(0)-1, bptt)):
        inputs, labels = get_batch(train_data, i)
        reshaped_inputs = inputs.view(3, 3, -1)
        print(reshaped_inputs)
        outputs, _ = net(reshaped_inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


"""
net = Net().to(device)
optimizer = optim.SGD(net.parameters(), lr=eta, momentum = momentum)
criterion = nn.CrossEntropyLoss()
for epoch in range(n_epochs):
    for i, data in enumerate(trainloader):
        inputs, labels = data.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
"""
#################################################### 

# Debugging




