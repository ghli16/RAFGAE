import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATConv
from utils import norm_adj

class resGAT(nn.Module):
    def __init__(self, in_feature, out_feature, num_layers, alpha, beta,
                 rna_feature, dis_feature):
        super(resGAT, self).__init__()
        self.num_layers = num_layers
        self.alpha = alpha
        self.beta = beta
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.rna_num = rna_feature
        self.input_fc = nn.Linear(in_feature, out_feature)
        self.output_xr = nn.Linear(out_feature, rna_feature)
        self.output_xd = nn.Linear(out_feature, dis_feature)

        for i in range(self.num_layers):
            self.convs.append(GATConv(out_feature, out_feature))
        self.weights = torch.nn.Parameter(torch.randn((len(self.convs))))

        self.att_c = Parameter(torch.ones((1, self.num_layers)), requires_grad=True)
        self.att_d = Parameter(torch.ones((1, self.num_layers)), requires_grad=True)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        self.input_fc.reset_parameters()
        self.output_xr.reset_parameters()
        self.output_xd.reset_parameters()
        torch.nn.init.normal_(self.weights)

    def forward(self, X, Adj):
        x = X
        x = self.input_fc(x)
        adj = Adj
        x_input = x

        layer_out = []
        xr = []
        xd = []
        for i in range(self.num_layers):
            x = self.convs[i](x, adj)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=0.5, training=self.training)
            if i == 0:
                x = x + self.alpha * x_input
            else:
                x = x + self.alpha * x_input + self.beta * layer_out[i - 1]
            layer_out.append(x)
            xr.append(
                torch.FloatTensor(x[:self.rna_num].clone()).float())
            xd.append(
                torch.FloatTensor(x[self.rna_num:].clone()).float())

        xr = [self.att_c[0][i] * xr[i] for i in range(len(xr))]
        xr = torch.sigmoid(self.output_xr(sum(xr)))

        xd = [self.att_d[0][i] * xd[i] for i in range(len(xd))]
        xd = torch.sigmoid(self.output_xd(sum(xd)))

        return xr, xd


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, drop=0.5, bias=False, activation=None):
        super(GraphConv, self).__init__()
        self.dropout = nn.Dropout(drop)
        self.activation = activation
        self.w = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.w.weight)
        self.bias = bias
        if self.bias:
            nn.init.zeros_(self.w.bias)

    def forward(self, adj, x):
        x = self.dropout(x)
        x = adj.mm(x)
        x = self.w(x)
        if self.activation:
            return self.activation(x)
        else:
            return x


class GAE(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(GAE, self).__init__()
        self.res1 = GraphConv(out_dim, hid_dim, activation=F.relu)
        self.res2 = GraphConv(hid_dim, out_dim, activation=torch.sigmoid)
    
    def forward(self, g, z):
        z = self.res1(g, z)
        res = self.res2(g, z)
        return res, z

class GATGAE(nn.Module):
    def __init__(self, hinencoder, drdecoder, didecoder):
        super(GATGAE, self).__init__()

        self.hinencoder = hinencoder
        self.drdecoder = drdecoder
        self.didecoder = didecoder

    def forward(self, x, adj, y0):
        xr, xd = self.hinencoder(x, adj)

        xr = xr.detach().numpy()
        xd = xd.detach().numpy()
        gr = norm_adj(xr)  # 标准化，作关联矩阵
        gd = norm_adj(xd.T)

        ydr, ydz = self.drdecoder(gr, y0)
        ydi, yiz = self.didecoder(gd, y0.T)
        return ydr, ydz, ydi, yiz