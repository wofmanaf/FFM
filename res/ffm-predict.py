# coding: utf-8
import argparse
from math import log

import dill
from FFM import sigmoid

import warnings

warnings.filterwarnings("ignore")

# arguments
parser = argparse.ArgumentParser(usage="ffm-predict.py test_file model_file output_file")
parser.add_argument('test_file')
parser.add_argument('model_file')
parser.add_argument('output_file')
args = parser.parse_args()

class Data:
    def __init__(self, lst):
        self.index = (lst[0], lst[1])
        self.value = float(lst[2])

if __name__ == '__main__':

    with open(args.model_file, mode='rb') as f:
        weights = dill.load(f)
        dimension = weights['dimension']

    with open(args.test_file, 'r') as f, open(args.output_file, 'w') as fw:
        ll = 0.
        n = 0
        for line in f:
            target, *lst = line.strip().split()
            feature = []
            for l in lst:
                d = Data(l.split(':'))
                feature.append(d)
            p = sigmoid(weights=weights, feature=feature)
            ll -= int(target) * log(p) + (1 - int(target)) * log(1 - p)
            n += 1

            target = 1 if target == '1' else -1
            fw.write('%s,%s\n' % (target, p))

    print('logloss: %.5f' % (ll / n))
