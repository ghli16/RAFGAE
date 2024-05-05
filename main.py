import argparse
from train import crossvalidation
import warnings
warnings.filterwarnings("ignore")
from scipy import interp
from utils import *
import seaborn
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed.')
parser.add_argument('--fold', type=int, default=10,
                    help='fold of cross-validation.')
parser.add_argument('--epochs', type=int, default=60,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Dimension of representations')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='Weight between drug space and disease space')
parser.add_argument('--alpha', type=float, default=0.3,
                    help='Weight of resGAT')
parser.add_argument('--beta', type=float, default=0.1,
                    help='Weight of resGAT')
parser.add_argument('--data', type=int, default=1, choices=[1, 2],
                    help='Dataset1')
parser.add_argument('--num_layers', type=int, default=4,
                    help='number of resGAT layers')
parser.add_argument('--FLAG', type=bool, default=True,
                    help='Disables FLAG training.')
parser.add_argument('--step-size', type=float, default=1e-3,
                    help='step size of adversarial perturbation.')
parser.add_argument('--m', type=int, default=2,
                    help='number of epochs to FLAG.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
set_seed(args.seed, args.cuda)


if __name__ == "__main__":
    dis_sim, drug_sim, ddi = load_data(args.data)
    print("Dataset {}, 10-fold CV".format(args.data))
    aucs, auprs = crossvalidation(ddi, dis_sim, drug_sim, args)

    mean_auc = sum(aucs)/args.fold
    mean_aupr = sum(auprs)/args.fold



    print("===Final result===")
    print('MEAN_AUC: %.4f | MEAN_AUPR: %.4f ' %
          (mean_auc, mean_aupr))
