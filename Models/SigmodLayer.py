import torch
from torch import nn
from torch.nn import init

class SigmodLayer(nn.Module):
    def __init__(self,embed_size,depth = 2,cuda =True):
        '''
        structure2vec中的 sig layer
        '''
        super(SigmodLayer, self).__init__()
        self.embed_size = embed_size
        self.depth = depth
        #self.cuda = cuda
        self.layers = []
        for i in range(self.depth):
            layer = nn.Linear(embed_size,embed_size,False)
            self.layers.append(layer)
            self.add_module('layer_{}'.format(i), layer)
    def forward(self, agg_embs):
        last_embed = agg_embs
        for i in range (self.depth):
            last_embed = self.layers[i](last_embed)
            if i != self.depth - 1:
                last_embed = torch.relu(last_embed)
        return last_embed