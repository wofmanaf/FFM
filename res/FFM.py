from collections import defaultdict
from itertools import chain
from copy import deepcopy
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def sigmoid(weights, feature):

    x = weights['w_0'] + sum([weights['w_i'][d.index] * d.value for d in feature])
    for j, d1 in enumerate(feature):
        for d2 in feature[j+1:]:
            x += np.dot(weights['v_ifk'][d1.index][d2.index[0]], weights['v_ifk'][d2.index][d1.index[0]]) * d1.value * d2.value

    return 1. / (1 + np.exp(-x))



def _Gradients(weights, instances):

    gradients = {'w_0': 0,
                 'w_i': defaultdict(int),
                 'v_ifk': defaultdict(lambda: defaultdict(lambda: np.zeros(weights['dimension'])))}

    for target, feature in instances:
        p = sigmoid(weights=weights, feature=feature)
        gradients['w_0'] += (p - target)
        for j, d in enumerate(feature):
            gradients['w_i'][d.index] += (p - target) * d.value
            for d2 in feature[j+1:]:
                gradients['v_ifk'][d.index][d2.index[0]] += (p - target) * (weights['v_ifk'][d2.index][d.index[0]] * d.value * d2.value)

    return gradients

def _Norm(weights, order):
    lst = list(chain.from_iterable([list(chain.from_iterable(v_if.values())) for v_if in weights['v_ifk'].values()]))
    vec = list(chain.from_iterable([[weights['w_0']], list(weights['w_i'].values()), lst]))
    return np.linalg.norm(vec, order)

def _Direction(gradients):

    norm = _Norm(weights=gradients, order=2)
    gradients['w_0'] /= - norm
    for index in gradients['w_i'].keys():
        gradients['w_i'][index] /= - norm
        for f in gradients['v_ifk'][index].keys():
            gradients['v_ifk'][index][f] /= - norm

    return gradients

def _LossFunc(weights, instances, regularization):

    loss = 0.5 * regularization * _Norm(weights=weights, order=2)
    for target, feature in instances:
        p = sigmoid(weights=weights, feature=feature)
        loss -= target * np.log(p) + (1 - target) * np.log(1 - p)
    return loss

def _Backtracking(weights, instances, direction, gradients, args):

    prior_weights = deepcopy(weights)

    # calculate loss
    prior_loss = _LossFunc(weights=prior_weights, instances=instances, regularization=args.regularization)

    # calculate a term of Armijo condition
    lst = list(chain.from_iterable([list(chain.from_iterable(v_if.values())) for v_if in direction['v_ifk'].values()]))
    d_vec = list(chain.from_iterable([[direction['w_0']], list(direction['w_i'].values()), lst]))
    lst = list(chain.from_iterable([list(chain.from_iterable(v_if.values())) for v_if in gradients['v_ifk'].values()]))
    g_vec = list(chain.from_iterable([[gradients['w_0']], list(gradients['w_i'].values()), lst]))
    term = args.control * np.dot(d_vec, g_vec)

    while 1:
        # calculate posterior weights
        weights['w_0'] = prior_weights['w_0'] + args.step_size * (direction['w_0'] - args.regularization * prior_weights['w_0'])
        for index in weights['w_i'].keys():
            weights['w_i'][index] = prior_weights['w_i'][index] + args.step_size * (direction['w_i'][index] - args.regularization * prior_weights['w_i'][index])
            for f in weights['v_ifk'][index].keys():
                weights['v_ifk'][index][f] = prior_weights['v_ifk'][index][f] + args.step_size * (direction['v_ifk'][index][f] - args.regularization * prior_weights['v_ifk'][index][f])

        # calculate posterior loss
        posterior_loss = _LossFunc(weights=weights, instances=instances, regularization=args.regularization)

        if posterior_loss <= prior_loss + args.step_size * term:
            return weights, posterior_loss / len(instances)
        else:
            args.step_size *= args.shrink

def _QuasiNewton(q, data_syc, H0, dimension):

    a_i = []
    a = {}
    for s, y, c in reversed(data_syc):
        a['w_0'] = c * s['w_0'] * q['w_0']
        q['w_0'] -= a['w_0'] * y['w_0']
        for index in q['w_i'].keys():
            a['w_i', index] = c * s['w_i', index] * q['w_i'][index]
            q['w_i'][index] -= a['w_i', index] * y['w_i', index]
            for f in q['v_ifk'][index].keys():
                for k in range(dimension):
                    a['v_ifk', index, f, k] = c * s['v_ifk', index, f, k] * q['v_ifk'][index][f][k]
                    q['v_ifk'][index][f][k] -= a['v_ifk', index, f, k] * y['v_ifk', index, f, k]

        a_i.insert(0, a)

    q['w_0'] *= H0
    for index in q['w_i'].keys():
        q['w_i'][index] *= H0
        for f in q['v_ifk'][index].keys():
            q['v_ifk'][index][f] *= H0

    for (s, y, c), a in zip(data_syc, a_i):
        b = c * y['w_0'] * q['w_0']
        q['w_0'] += s['w_0'] * (a['w_0'] - b)
        for index in q['w_i'].keys():
            b = c * y['w_i', index] * q['w_i'][index]
            q['w_i'][index] += s['w_i', index] * (a['w_i', index] - b)
            for f in q['v_ifk'][index].keys():
                for k in range(dimension):
                    b = c * y['v_ifk', index, f, k] * q['v_ifk'][index][f][k]
                    q['v_ifk'][index][f][k] += s['v_ifk', index, f, k] * (a['v_ifk', index, f, k] - b)

    return q

def LBFGS(tr_instances, va_instances, weights, args):

    data_syc = []
    s = {}
    y = {}

    # calculate initial gradients
    g = _Gradients(weights=weights, instances=tr_instances)

    print('iter\ttr_loss\tva_loss')
    for iter in range(1, args.iterations + 1):

        prev_weights = deepcopy(weights)
        prev_gradients = deepcopy(g)

        # calculate inverse hessian matrix
        q = deepcopy(g)
        if iter > 1:
            q = _QuasiNewton(q=q, data_syc=data_syc, H0=H0, dimension=weights['dimension'])

        # search direction
        d = _Direction(gradients=q)

        # backtracking
        weights, tr_loss = _Backtracking(weights=weights, instances=tr_instances, direction=d, gradients=g, args=args)

        # calculate gradients
        g = _Gradients(weights=weights, instances=tr_instances)

        # store deltas
        s['w_0'] = weights['w_0'] - prev_weights['w_0']
        y['w_0'] = g['w_0'] - prev_gradients['w_0']
        for index in weights['w_i'].keys():
            s['w_i', index] = weights['w_i'][index] - prev_weights['w_i'][index]
            y['w_i', index] = g['w_i'][index] - prev_gradients['w_i'][index]
            for f in weights['v_ifk'][index].keys():
                for k in range(weights['dimension']):
                    s['v_ifk', index, f, k] = weights['v_ifk'][index][f][k] - prev_weights['v_ifk'][index][f][k]
                    y['v_ifk', index, f, k] = g['v_ifk'][index][f][k] - prev_gradients['v_ifk'][index][f][k]

        yts = np.dot(list(y.values()), list(s.values()))
        c = 1. / yts
        data_syc.append((s, y, c))

        # calculate H0
        yty = np.dot(list(y.values()), list(y.values()))
        H0 = yts / yty

        # limit memory
        if len(data_syc) > args.memory:
            del data_syc[0]

        # calculate logloss of validation dataset
        va_loss = _LossFunc(weights=weights, instances=va_instances, regularization=0) / len(va_instances)
        print('%d\t%.5f\t%.5f' % (iter, tr_loss, va_loss))

        # convergence test
        # if iter > 1 and (prev_va_loss < va_loss or prev_tr_loss - tr_loss < args.threshold):
        #     print('Auto-stop. Use model at %dth iteration.' % (iter - 1))
        #     return prev_weights
        if iter == args.iterations:
            print('Use model at %dth iteration.' % (iter))
            return weights
        else:
            prev_tr_loss = tr_loss
            prev_va_loss = va_loss
