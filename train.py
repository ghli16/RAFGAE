from models import resGAT, GAE, GATGAE
from utils import *
from attacks import FLAG
from sklearn.preprocessing import minmax_scale
import torch.nn.functional as F

def train(Model, y0, epoch, gamma, X, Adj, args):
    opt = torch.optim.Adam(Model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for e in range(epoch):
        if args.FLAG:
            def forward(perturb):
                return Model(X + perturb, Adj, y0)
            model_forward = (Model, forward)
            loss, yl, yd = FLAG(model_forward, X.shape, y0, gamma, opt, args)
        else:
            Model.train()
            opt.zero_grad()
            yl, zl, yd, zd = Model(X, Adj, y0)
            losspl = F.binary_cross_entropy(yl, y0)
            losspd = F.binary_cross_entropy(yd, y0.t())
            loss = gamma * losspl + (1 - gamma) * losspd
            loss.backward()
            opt.step()
            Model.eval()
            with torch.no_grad():
                yl, _, yd, _ = Model(X, Adj, y0)

        print('Epoch %d | Lossp: %.4f' % (e, loss.item()))
    return gamma * yl + (1 - gamma) * yd.t()


def crossvalidation(A, dis_sim, drug_sim, args):
    N = A.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    res = torch.zeros(args.fold, A.shape[0], A.shape[1])
    drug_num = A.shape[0]
    dis_num = A.shape[1]
    aucs = []
    auprs = []


    for i in range(args.fold):
        print("Fold {}".format(i+1))
        A0 = A.clone()
        for j in range(i*N//10, (i+1)*N//10):
            A0[idx[j], :] = torch.zeros(A.shape[1])

        Heter_adj = constructNet(A0)
        Heter_adj = norm_adj(Heter_adj)
        Heter_adj = torch.FloatTensor(Heter_adj)
        Heter_adj_edge_index = get_edge_index(Heter_adj)
        Adj = Heter_adj_edge_index

        X = constructHNet(A0, dis_sim, drug_sim)
        X = minmax_scale(X, axis=0)
        X = torch.FloatTensor(X)


        Model = GATGAE(resGAT(X.shape[0], args.hidden, args.num_layers, args.alpha, args.beta,
                        drug_num, dis_num), GAE(A.shape[0], A.shape[1]), GAE(A.shape[1], A.shape[0]))
        resi = train(Model, A0, args.epochs, args.gamma, X, Adj, args)

        resi = scaley(resi)
        res[i] = resi
        resi = resi.detach().numpy()
        np.savetxt('results.csv', resi, delimiter=',', fmt='%2f')
        auroc, aupr = show_auc(resi, args.data)

        aucs.append(auroc)
        auprs.append(aupr)


    return aucs, auprs
