import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = input @ (self.weight.repeat(input.shape[0],1,1))    # X * W
        output = adj @ support           # A * X * W
        if self.bias is not None:        # A * X * W + b
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(256, 64)
        #self.gc1 = GraphConvolution(32, 16)
        self.ln1 = nn.LayerNorm(64)
        self.gc2 = GraphConvolution(64, 16)
        self.ln2 = nn.LayerNorm(16)
        self.relu1 = nn.LeakyReLU(0.2,inplace=True)
        self.relu2 = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x, adj):  			# x.shape = (seq_len, GCN_FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x = self.gc1(x, adj)  				# x.shape = (seq_len, GCN_HIDDEN_DIM)
        x = self.relu1(self.ln1(x))
        x = self.gc2(x, adj)
        output = self.relu2(self.ln2(x))	# output.shape = (seq_len, GCN_OUTPUT_DIM)
        return output

class Attention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.n_heads)

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input):  				# input.shape = (1, seq_len, input_dim)
        x = self.fc1(input)  	            # x.shape = (1, seq_len, attention_hops)
        x = self.softmax(x, 1)  					
        attention = x.transpose(1, 2)  		# attention.shape = (1, attention_hops, seq_len)
        return attention


class Model(nn.Module):
    def __init__(self):
        super(Model5, self).__init__()
        self.gcn = GCN()
        self.attention = Attention(16, 4)
        self.fc1 = nn.Linear(2048, 256)
        #self.fc1 = nn.Linear(108, 64)
        self.fc2 = nn.Linear(24, 16)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)
        self.fc_final = nn.Linear(16, 1)
        self.activate = torch.nn.Tanh()
        self.criterion = nn.MSELoss()
        #self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.001)

    def forward(self, x1, adj,attention): 	
        adj = adj.float()		
        x1 = self.dropout(x1)							# x.shape = (batch,seq_len, FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x1 = self.fc1(x1)
        x = self.gcn(x1, adj)								# x.shape = (seq_len, GAT_OUTPUT_DIM)
        x2 = torch.cat((x,attention),dim=2)
        x2 = self.fc2(x2)
        att = self.attention(x2)  											# att.shape = (1, ATTENTION_HEADS, seq_len)
        node_feature_embedding = att @ x2								# output.shape = (1, ATTENTION_HEADS, GAT_OUTPUT_DIM)
        node_feature_embedding_avg = torch.sum(node_feature_embedding,1) / att.shape[1]  # node_feature_embedding_avg.shape = (1, GAT_OUTPUT_DIM)
        
        o = self.fc_final(node_feature_embedding_avg)
        output = self.activate(o)  	# output.shape = (1, NUM_CLASSES)
        return output