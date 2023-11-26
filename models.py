import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


class IQN(nn.Module):
    def __init__(self, state_size, action_size, layer_size, n_step, seed, N, device="cuda:0"):
        super(IQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.state_dim = len(self.input_shape)
        self.action_size = action_size
        self.N = N  
        self.n_cos = 64
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi*i for i in range(1,self.n_cos+1)]).view(1,1,self.n_cos).to(device) # Starting from 0 as in the paper 
        self.device = device
        layer = nn.Linear

        # Network Architecture
        if self.state_dim == 3:
            self.head = nn.Sequential(
                nn.Conv2d(4, out_channels=32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            )#.apply() #weight init
            self.cos_embedding = nn.Linear(self.n_cos, self.calc_input_layer())
            self.ff_1 = layer(self.calc_input_layer(), layer_size)
            self.cos_layer_out = self.calc_input_layer()

        else:
            self.head = nn.Linear(self.input_shape[0], layer_size) 
            self.cos_embedding = nn.Linear(self.n_cos, layer_size)
            self.ff_1 = layer(layer_size, layer_size)
            self.cos_layer_out = layer_size
        
        self.ff_2 = layer(layer_size, action_size)    

    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.head(x)
        return x.flatten().shape[0]
        
    def calc_cos(self, batch_size, n_tau=8):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1).to(self.device) 
        cos = torch.cos(taus*self.pis)

        assert cos.shape == (batch_size,n_tau,self.n_cos), "cos shape is incorrect"
        return cos, taus
    
    def forward(self, input, num_tau=8):
        """
        Quantile Calculation depending on the number of tau
        
        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]
        
        """
        batch_size = input.shape[0]
        
        x = torch.relu(self.head(input))
        if self.state_dim == 3: x = x.view(input.size(0), -1)
        cos, taus = self.calc_cos(batch_size, num_tau) # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size*num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.cos_layer_out) # (batch, n_tau, layer)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1)*cos_x).view(batch_size*num_tau, self.cos_layer_out)
        
        x = torch.relu(self.ff_1(x))
        
        out = self.ff_2(x)
        
        return out.view(batch_size, num_tau, self.action_size), taus
    
    def get_qvalues(self, inputs):
        quantiles, _ = self.forward(inputs, self.N)
        actions = quantiles.mean(dim=1)
        return actions  
    
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from copy import copy


def weight_init_xavier(layers):
    for layer in layers:
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)

class QVN(nn.Module):
    """Quantile Value Network"""
    def __init__(self, state_size, action_size,layer_size, n_step, device, seed, N=32):
        super(QVN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.state_dim = len(self.input_shape)
        self.N = N
        self.n_cos = 64
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi*i for i in range(1, self.n_cos+1)]).view(1,1,self.n_cos).to(device) # Starting from 0 as in the paper 
        self.device = device

        layer = nn.Linear


        # Network Architecture
        if self.state_dim == 3:
            self.head = nn.Sequential(
                nn.Conv2d(4, out_channels=32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            )#.apply() #weight init
            self.cos_embedding = layer(self.n_cos, self.calc_input_layer())
            self.ff_1 = layer(self.calc_input_layer(), layer_size)
            self.cos_layer_out = self.calc_input_layer()

        else:   
            self.head = nn.Linear(self.input_shape[0], layer_size) 
            self.cos_embedding = nn.Linear(self.n_cos, layer_size)
            self.ff_1 = layer(layer_size, layer_size)
            self.cos_layer_out = layer_size
            weight_init([self.head, self.ff_1])

        self.ff_2 = layer(layer_size, action_size)    
        weight_init([self.ff_1])
            
    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.head(x)
        return x.flatten().shape[0]
        
    def calc_cos(self,taus):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        batch_size = taus.shape[0]
        n_tau = taus.shape[1]
        cos = torch.cos(taus.unsqueeze(-1)*self.pis)

        assert cos.shape == (batch_size,n_tau,self.n_cos), "cos shape is incorrect"
        return cos
    
    def forward(self, input):
        """Calculate the state embeddings"""
        if self.state_dim == 3:
            x =  torch.relu(self.head(input))
            return x.view(input.size(0), -1)
        else:
            return torch.relu(self.head(input))
        
    def get_quantiles(self, input, taus, embedding=None):
        """
        Quantile Calculation depending on the number of tau
        
        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        
         """
        if embedding==None:
            x = self.forward(input)
            if self.state_dim == 3: x = x.view(input.size(0), -1)
        else:
            x = embedding
        batch_size = x.shape[0]
        num_tau = taus.shape[1]
        cos = self.calc_cos(taus) # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size*num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.cos_layer_out) # (batch, n_tau, layer)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1)*cos_x).view(batch_size*num_tau, self.cos_layer_out)   
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        return out.view(batch_size, num_tau, self.action_size)
    
    

class FPN(nn.Module):
    """Fraction proposal network"""
    def __init__(self, layer_size, seed, num_tau=8, device="cuda:0"):
        super(FPN,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.num_tau = num_tau
        self.device = device
        self.ff = nn.Linear(layer_size, num_tau)
        self.softmax = nn.LogSoftmax(dim=1)
        weight_init_xavier([self.ff])
        
    def forward(self,x):
        """
        Calculates tau, tau_ and the entropy
        
        taus [shape of (batch_size, num_tau)]
        taus_ [shape of (batch_size, num_tau)]
        entropy [shape of (batch_size, 1)]
        """

        q = self.softmax(self.ff(x)) 
        q_probs = q.exp()
        taus = torch.cumsum(q_probs, dim=1)
        taus = torch.cat((torch.zeros((q.shape[0], 1)).to(self.device), taus), dim=1)
        taus_ = (taus[:, :-1] + taus[:, 1:]).detach() / 2.
        
        entropy = -(q * q_probs).sum(dim=-1, keepdim=True)
        assert entropy.shape == (q.shape[0], 1), "instead shape {}".format(entropy.shape)
        
        return taus, taus_, entropy