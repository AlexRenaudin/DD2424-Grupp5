# Import
import numpy as np
import torch
import torch.nn as nn
import torchtext
from torchtext.datasets import PennTreebank
from scipy.special import softmax

# GPU


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



class ModelInstance():

    def __init__(self, variant = 'rnn'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.variant = variant
        print(f'Running variant: {variant}')
        # Parameters
        self.n_epochs = 1
        self.training_losses = []
        self.test_losses = []
        self.eta = 0.001
        self.alpha = 0.9
        self.seq_len  = 25
        self.m = 100
        #variant = 'rnn' #rnn, lstm, 2-lstm, bi-lstm, gru
        self.layerxdirection = 1
        if self.variant == '2-lstm' or self.variant == 'bi-lstm':
            self.layerxdirection = 2
        self.use_convnet = False
        self.convnet_out_channels = 5
        self.use_torchdata = True
        #file_name = 'names.txt'
        #self.file_name = 'goblet_book.txt'
        #self.file_name = 'goblet_short.txt'
        #file_name = 'Hermione.txt'
        #file_name = 'random.txt'



        if self.use_torchdata:
            self.train_unique_text, self.test_unique_text = PennTreebank(split=('train','test'))
            self.train_count, self.unique = unique_count(self, self.train_unique_text)
            self.test_count, _ = unique_count(self, self.test_unique_text)
        else:
            self.unique_text = open(self.file_name,'r').readlines()
            self.full_count, self.unique = unique_count(self, unique_text)
    


        self.int2char = {}
        self.char2int = {}
        self.K = 0
        for c in self.unique:
            self.int2char[self.K] = c
            self.char2int[c] = self.K
            self.K += 1
        

    def run_model(self):
        print('')
        try:
            train_tensor = torch.load('train_tensor.pt')
            test_tensor = torch.load('test_tensor.pt')
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
            if self.use_torchdata:
                train_text, test_text = PennTreebank(split=('train', 'test'))
                train_tensor = text2tensor((self.train_count//self.seq_len)+1, train_text, self.seq_len, self.char2int)
                test_tensor = text2tensor((self.test_count//self.seq_len)+1, test_text, self.seq_len, self.char2int)
            else:
                full_text = open(self.file_name,'r').readlines()
                n_seq = (self.full_count//self.seq_len)+1
                full_tensor = text2tensor(n_seq, full_text, self.seq_len, self.char2int)
                test_size = n_seq//10
                train_tensor = full_tensor[:,0:n_seq-test_size]
                test_tensor = full_tensor[:,n_seq-test_size:n_seq]        
        torch.save(train_tensor, 'train_tensor.pt')   
        torch.save(test_tensor, 'test_tensor.pt')

        net = Net(self.K, self.m, self.variant, self.use_convnet, self.convnet_out_channels).to(self.device)
        optimizer = torch.optim.RMSprop(net.parameters(), lr=self.eta, alpha = self.alpha)
        criterion = nn.CrossEntropyLoss()


        for epoch in range(self.n_epochs):
            H = torch.zeros(self.layerxdirection, 1, self.m).to(self.device)
            C = torch.zeros(self.layerxdirection, 1, self.m).to(self.device)
            for i in range(train_tensor.size(1)-1):
                X = train_tensor[:,i]
                target_Y = torch.cat((train_tensor[1:self.seq_len,i],train_tensor[0:1,i+1]),0).long()
                forward_Y, H, C = net(encode(self, X, self.K)[:,None,:], H, C)
                loss = criterion(forward_Y, target_Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i == 0:
                    smooth_loss = loss.item()
                else:
                    smooth_loss = 0.999* smooth_loss + 0.001*loss.item()  
                    self.training_losses.append(smooth_loss)          
                if i/100 == i//100: 
                    print(f'training iteration {i}/{train_tensor.size(1)-1} epoch {epoch+1}/{self.n_epochs} for {self.variant}', end = '\r')
                    #print('smooth loss =', smooth_loss)
                    pass
                if i/100 == i//100: 
                    #text = generate(self, H, C, 200, self.char2int, self.int2char, self.K, net)
                    #print(text)
                    pass
        print('')
        for i in range(test_tensor.size(1)-1):
            X = test_tensor[:,i]
            target_Y = torch.cat((test_tensor[1:self.seq_len,i],test_tensor[0:1,i+1]),0).long()
            forward_Y, H, C = net(encode(self,X, self.K)[:,None,:], H, C)
            loss = criterion(forward_Y, target_Y)
            loss.backward()
            if i == 0:
                smooth_loss = loss.item()
            else:
                smooth_loss = 0.999 * smooth_loss + 0.001*loss.item()
                self.test_losses.append(smooth_loss)
            if i/100 == i//100: 
                print(f'test iteration {i}/{test_tensor.size(1)-1} for {self.variant}', end = '\r')
                #print('test loss =', smooth_loss)
        #text = generate(self, H, C, 1000, self.char2int, self.int2char, self.K, net)
        print('')
    
    # Unique Characters
    def get_losses(self):
        return self.training_losses, self.test_losses
def unique_count(self, unique_text):
    unique = {}
    c_count = 0
    for line in unique_text:
        for word in line:
            for c in word:
                unique[c] = True
                c_count += 1
    return c_count, unique

def encode(self, tensor, K):
    n = tensor.size()[0]
    new_tensor = torch.zeros(n, K)
    for i in range(n):
        new_tensor[i,tensor[i].long()] = 1 
    return new_tensor

# Text generator
def generate(self,H, C, n_gen, char2int, int2char, K, net):
    X_gen = torch.tensor([char2int['a']])
    X_gen = encode(self,X_gen, K)
    text_gen = ''
    H_gen = H
    C_gen = C
    for i in range(n_gen):
        y_gen, H_gen, C_gen = net(X_gen[:,None,:], H_gen, C_gen)
        p_gen = y_gen[0].detach().numpy()
        #p_gen =  np.exp(p_gen)/sum(np.exp(p_gen))
        p_gen = softmax(p_gen)
        sample = np.random.choice(a = range(K), p = p_gen)
        text_gen = text_gen + int2char[sample]
        X_gen = y_gen
    return text_gen

def run_model(variant):
    m = ModelInstance(variant)
    m.run_model()
    test_l, test_l = m.get_losses()
    return test_l, test_l
