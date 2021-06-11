import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
torch.manual_seed(0)
torch.cuda.manual_seed(0)

class GraphConvolutionLayer(Module):

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / torch.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(Module):
    def __init__(self, nfeat, emb_dim, nvocab, sent_len, dropout):
        super(GCN, self).__init__()

        self.dep_embs = torch.nn.Embedding(nfeat, emb_dim)

        self.gc1 = GraphConvolutionLayer(emb_dim, nvocab)


    def forward(self, x, adj):
        dep_matrix = self.dep_embs(x.long())
        h1 = torch.matmul(adj, dep_matrix)
        act_h1 = F.relu(h1)
        y = self.gc1(act_h1, adj)
        return y
