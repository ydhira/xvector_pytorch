from __future__ import division
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.autograd import Variable
import sys

class XVector(nn.Module):
    def __init__(self, input_vector_length,  filter_sizes, kernel_sizes, num_classes,input_dim,embeded_sizes):
        super(XVector, self).__init__()
        layers = nn.ModuleList()
        embedding_layer_a = nn.ModuleList()
        embedding_layer_b = nn.ModuleList()
        prev_dim = input_dim
        for i, (kernel_size,filter_size) in enumerate(zip(kernel_sizes,filter_sizes)):
            print("prev_dim= %d, filter_size= %d, kernel_size= %d" \
                %(prev_dim, filter_size, kernel_size))
            conv = nn.Conv1d(in_channels= prev_dim, out_channels= filter_size, kernel_size= kernel_size, stride=1, bias=True)
            conv.bias.data.fill_(0.1)
            conv.weight.data.normal_(0, 0.1)
            bn = nn.BatchNorm1d(filter_size)
            relu = nn.ReLU()
            prev_dim = filter_size
            layers.extend([conv, bn, relu])
            if i != len(kernel_sizes)-1:
                layers.append(nn.Dropout(0.5))

        self.layers = nn.Sequential( *layers)
        self.layerM = layers 
        prev_dim = 2 * prev_dim

        for i, out_dim in enumerate(embeded_sizes):
            fc = nn.Linear(prev_dim, out_dim)
            fc.bias.data.fill_(0.1)
            fc.weight.data.normal_(0, 0.1)
            bn = nn.BatchNorm1d(out_dim)
            relu = nn.ReLU()
            prev_dim = out_dim
            if i==0: 
                self.embedding_a = fc
                embedding_layer_a.extend([bn, relu])

            if i==1: 
                self.embedding_b = fc
                embedding_layer_b.extend([bn, relu])
            
        self.embedding_layer_a = nn.Sequential( *embedding_layer_a)
        self.embedding_layer_b = nn.Sequential( *embedding_layer_b)

        # self.embedding_layerM = embedding_layer
        self.fc1 = nn.Linear(prev_dim, num_classes)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.1)

    def forward(self, x):
        out = self.layers(x)
        ## stats pooling layer 
        ## a statistics pooling layer that aggregates
        ## over the frame-level representations
        mean, var = torch.mean(out, dim=2), torch.var(out, dim=2)
        out = torch.cat([mean, var], dim=1)
        embed_a_out = self.embedding_a(out)
        out = self.embedding_layer_a(embed_a_out)
        out = nn.Dropout(0.5)(out)
        embed_b_out = self.embedding_b(out)
        out = self.embedding_layer_b(embed_b_out)
        out = self.fc1(out)
        out = F.log_softmax(out, dim=1)
        return embed_a_out, embed_b_out, out