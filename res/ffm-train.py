# coding: utf-8
import argparse
from collections import defaultdict

import dill
import numpy as np
from FFM import LBFGS
import warnings

warnings.filterwarnings("ignore")
# arguments
parser = argparse.ArgumentParser(usage='ffm-train.py [options] training_set_file validation_set_file model_file')
parser.add_argument('train_set_file')
parser.add_argument('validation_set_file')
parser.add_argument('model_file')
group_train = parser.add_argument_group('training options')
group_train.add_argument('--iter', metavar='int', dest='iterations', type=int, default=100,
                         help='number of iterations (default 1000)')
group_train.add_argument('--threshold', metavar='float', dest='threshold', type=int, default=1e-8,
                         help='termination threshold (default 1e-6)')
group_train.add_argument('--memory', metavar='int', dest='memory', type=int, default=10,
                         help='number of limited memory matrix (default 10)')
group_model = parser.add_argument_group('model params')
group_model.add_argument('-k', '--factor', metavar='int', dest='factors', type=int, default=10,
                         help='number of latent factors (default 10)')
group_model.add_argument('--sigma', metavar='float', dest='sd', type=float, default=0.01,
                         help='standard deviation σ > 0 (default 0.01)')
group_model.add_argument('--mu', metavar='float', dest='regularization', type=float, default=2e-5,
                         help='regularization parameter μ > 0 (default 2e-5)')
group_model.add_argument('--alpha', metavar='float', dest='step_size', type=float, default=1.0,
                         help='step size parameter α > 0 (default 1.0)')
group_model.add_argument('--c', metavar='float', dest='control', type=float, default=0.5,
                         help='control parameter c ∈ (0,1) (default 0.5)')
group_model.add_argument('--rho', metavar='float', dest='shrink', type=float, default=0.5,
                         help='shrink parameter ρ ∈ (0,1) (default 0.5)')
args = parser.parse_args()

class Data:
    def __init__(self, lst):
        self.index = (lst[0], lst[1])
        self.value = float(lst[2])

if __name__ == '__main__':

    # init weights
    dimension = args.factors
    weights = {'dimension': dimension,
               'w_0': 0,
               'w_i': defaultdict(int),
               'v_ifk': defaultdict(lambda: defaultdict(lambda: np.zeros(dimension)))}

    # read train set file
    tr_instances = []
    with open(args.train_set_file, 'r') as f:
        for line in f:
            target, *lst = line.strip().split()
            feature = []
            for l in lst:
                d = Data(l.split(':'))
                feature.append(d)

            for j, d1 in enumerate(feature):
                for d2 in feature[j+1:]:
                    weights['v_ifk'][d1.index][d2.index[0]] = np.random.normal(0, args.sd, dimension)

            tr_instances.append((int(target), feature))

    # read validation set file
    va_instances = []
    with open(args.validation_set_file, 'r') as f:
        for line in f:
            target, *lst = line.strip().split()
            feature = []
            for l in lst:
                d = Data(l.split(':'))
                feature.append(d)

            va_instances.append((int(target), feature))

    # start training
    weights = LBFGS(tr_instances=tr_instances, va_instances=va_instances, weights=weights, args=args)

    # write model file
    dill.dump(weights, open(args.model_file, mode='wb'))
