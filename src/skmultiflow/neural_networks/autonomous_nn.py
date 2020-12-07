from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils.utils import *
from skmultiflow.neural_networks.utils import *

import os
import random
import copy
import builtins
import time
import numpy as np
import scipy.io
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="input file or stream", default='/Users/ng98/Desktop/datasets/unziped/forExperiments/nomao_class_label_mapped.csv')
parser.add_argument("-o", "--output", type=str, help="output file", default='out.csv')
parser.add_argument("-b", "--batchsize", type=int, help="Batch Size", default=200)
args = parser.parse_args()
if args.input is None:
    print('Input file or stream is not given')
    exit(1)

# Write-Overwrites
# file1 = open(args.output, "w", buffering=1024*512)  # write mode
# file1.write("id,mean_acc_[M0],current_acc_[M0],mean_kappa_[M0],current_kappa_[M0],training_time_[M0],testing_time_[M0],total_running_time_[M0],model_size_[M0]\n")

print(args.batchsize)

if args.batchsize < 100:
    println = builtins.print

    def print(*args, **kwargs):
            # builtins.print(*args, **kwargs)
            pass


# Notebook code-----------------------------------------------------------------------------------
class nn(object):
    def __init__(self, layer):
        self.size = layer
        self.n = len(layer)
        self.hl = self.n - 2
        self.activation_func = "sigm"
        self.learningRate = 0.01
        self.momentum = 0.95
        self.outputConnect = 1
        self.output = "softmax"

        # initiate weights and weight momentum for hidden layer
        self.W = {}
        self.vW = {}
        self.dW = {}
        self.c = {}
        # self.W = self.vW  = self.dW= self.c = {}
        for i in range(1, self.n - 1):
            self.W[i] = np.random.normal(0, np.sqrt(2 / (self.size[i - 1] + 1)),
                                         size=(self.size[i], self.size[i - 1] + 1))
            self.vW[i] = np.zeros(self.W[i].shape)
            self.dW[i] = np.zeros(self.W[i].shape)
            self.c[i] = np.random.normal(0, np.sqrt(2 / (self.size[i - 1] + 1)), size=(self.size[i - 1], 1))

        # initiate weights and weight momentum for output layer
        self.Ws = {}
        self.vWs = {}
        self.dWs = {}
        self.beta = {}
        self.betaOld = {}
        self.p = {}
        if self.outputConnect == 1:
            for i in range(1, (self.hl + 1)):
                self.Ws[i] = np.random.normal(0, np.sqrt(2 / ((self.W[i].shape[0]) + 1)),
                                              size=(self.size[len(self.size) - 1], self.size[i] + 1))
                self.vWs[i] = np.zeros(self.Ws[i].shape)
                self.dWs[i] = np.zeros(self.Ws[i].shape)
                self.beta[i] = 1
                self.betaOld[i] = 1
                self.p[i] = 1
        else:
            i = 1
            self.Ws[i] = np.random.normal(0, np.sqrt(2 / ((self.W[i - 1].shape[0]) + 1)),
                                          size=(self.size[len(self.size) - 1], self.size[len(self.size) - 2] + 1))
            self.vWs[i] = np.zeros(self.Ws[i].shape)
            self.dWs[i] = np.zeros(self.Ws[i].shape)

            # initiate later used variable
        self.a = {}
        self.aas = {}
        self.e = {}
        self.L = {}
        self.classlabel = {}
        self.nop = {}
        self.nodes = {}


class Parameter(object):
    def __init__(self, nn,layer,K):
        self.nn = nn
        self.ev = {}
        self.size = layer
        self.prune_list       = 0
        self.prune_list_index = []
        self.ev[1] = {'layer': layer, 'kp':0, 'kl':0 ,'miu_x_old':0, 'var_x_old':0, 'kl':0,'K':K, 'cr':0,'node':{},
                     'BIAS2':{}, 'VAR':{}, 'miu_NS_old':0, 'var_NS_old':0, 'miu_NHS_old':0, 'var_NHS_old':0,
                     'miumin_NS':[], 'miumin_NHS':[], 'stdmin_NS':[], 'stdmin_NHS':[]}
        self.Loss = {}
        self.cr = {}
        self.wl = {}



class Performance(object):
    def __init__(self):
        self.update_time = 0
        self.ev = {}
        self.test_time = 0
        self.classification_time    = 0
        self.layer = 0
        self.ev[1] = {'f_measure': 0, 'g_mean':0, 'recall':0, 'precision':0}
        self.LayerWeight = 0
        self.meanode = []
        self.stdnode = []
        self.NumberOfParameters = 0


def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))


def relu_array(x):
    return abs(x) * (x > 0)


def softmax_array(x):
    list_re = []
    for arr in x:
        e_x = np.exp(arr - np.max(arr))
        result = e_x / e_x.sum()
        list_re.append(result)
    return np.array(list_re)


def argmax_array(x):
    list_re = []
    for arr in x:
        result = [arr.argmax()]
        list_re.append(result)
    return np.array(list_re)


def valuemax_array(x):
    list_re = []
    for arr in x:
        result = [arr.max()]
        list_re.append(result)
    return np.array(list_re)


def update_beta(x):
    result = {}
    sum_val = sum(x.values())
    #     if(all(value == 0 for value in x.values())):
    #         return x

    for k, v in x.items():
        if sum_val == 0:
            result[k] = 0.0
        else:
            result[k] = v / sum_val
    return result


def checkbeta(dict1):
    count = 0
    for i in dict1.values():
        if (i != 0):
            count += 1
    return count


def nettestparallel(nn, x, y, ev):
    nop = []
    # feedforward
    nn = netfeedforward(nn, x, y)
    (m1, m2) = y.shape
    compare0 = 0
    comparenot0 = 0
    # obtain true class
    act = argmax_array(y)
    for i in range(1, nn.hl + 1):
        nn.classlabel[i] = {}
    if (len(nn.betaOld) > 1):
        nn.sigma = np.zeros((m1, m2))
        for t in range(0, m1):
            for i in range(1, nn.hl + 1):
                if (nn.beta[i] != 0):
                    # obtain predicted label, note : layer weight betaOld is fixed
                    nn.sigma[t] = nn.sigma[t] + (nn.aas[i][t] * nn.betaOld[i])
                    nn.classlabel[i][t] = nn.aas[i][t].argmax()
                    compare = act[t][0] - nn.classlabel[i][t]
                    # train the weighted voting
                    if (compare != 0):
                        nn.beta[i] = max(nn.beta[i] * nn.p[i], 0)
                        nn.p[i] = max(nn.p[i] - 0.01, 0)
                    elif (compare == 0):
                        nn.beta[i] = min(nn.beta[i] * (1 + nn.p[i]), 1)
                        nn.p[i] = min(nn.p[i] + 0.01, 1)

                # last element on data chunk
                if (t == m1):
                    # calculate number of parameter
                    if (nn.beta[i] != 0):
                        (c, d) = nn.Ws[i].shape
                        vw = 1
                    else:
                        c = 0
                        d = 0
                        vw = 0
                    (a, b) = nn.W[i].shape
                    nop.append(a * b + c * d + vw)

                    # calculate the number of node in each hidden layer
                    nn.nodes[i][nn.t] = ev[1]['K']
            # print(nn.beta)
            nn.beta = update_beta(nn.beta)
            # print(nn.beta)
    elif (len(nn.betaOld) == 1):
        nn.sigma = nn.aas[1]
        (c, d) = nn.Ws[1].shape
        vw = 1
        (a, b) = nn.W[1].shape
        nop = a * b + c * d + vw
        nn.nodes[1] = {nn.t: ev[1]['K']}

    nn.nop[nn.t] = np.sum(nop)

    nn.mnop = [np.array(list(nn.nop.values())).mean(), np.array(list(nn.nop.values())).std()]

    # update voting weight
    nn.betaOld = nn.beta.copy()
    nn.index = max(nn.beta, key=nn.beta.get)

    # calculate classification rate
    [raw_out, out] = [valuemax_array(nn.sigma), argmax_array(nn.sigma)]
    nn.bad = (np.where(out != act)[0]).reshape(-1, 1)
    nn.cr = 1 - (nn.bad.shape[0] / m1)
    nn.residual_error = 1 - raw_out
    nn.out = out
    nn.act = act
    return [nn]
    # return nn,out,act,raw_out


def netfeedforward(nn, x, y):
    n = nn.n
    m = x.shape[0]
    ones = np.ones(shape=(m, 1))
    x = np.append(ones, x, axis=1)  # append columns of 1 in matrix for bias
    nn.a[1] = x
    # feedforward from input layer through all the hidden layer
    for i in range(2, n):
        if (nn.activation_func == 'sigm'):
            nn.a[i] = sigmoid_array(np.matmul(nn.a[i - 1], nn.W[i - 1].T))
        elif (nn.activation_func == 'relu'):
            nn.a[i] = relu_array(np.matmul(nn.a[i - 1], nn.W[i - 1].T))
        ones = np.ones(shape=(m, 1))
        nn.a[i] = np.append(ones, nn.a[i], axis=1)

    # propagate to the output layer
    for i in range(1, nn.hl + 1):
        if nn.beta[i] != 0:
            if (nn.output == 'sigm'):
                nn.aas[i] = sigmoid_array(np.matmul(nn.a[i + 1], nn.Ws[i].T))
            elif (nn.output == 'linear'):
                nn.aas[i] = np.matmul(nn.a[i + 1], nn.Ws[i].T)
            elif (nn.output == 'softmax'):
                nn.aas[i] = softmax_array(np.matmul(nn.a[i + 1], nn.Ws[i].T))

                # calculate error
            nn.e[i] = y - nn.aas[i]

            # calculate loss function
            if (nn.output == 'sigm' or nn.output == 'linear'):
                nn.L[i] = 0.5 * (np.sum(np.sum(nn.e[i] ** 2))) / m
            elif (nn.output == 'softmax'):
                nn.L[i] = -np.sum(np.sum(y * np.log(nn.aas[i]))) / m

            # print(nn.L)
    return nn


#net configuration while training
def netconfigtrain(layer):
    net = nn(layer)
    net.layer = layer
    net.n = len(layer)
    net.activation_func = "sigm"
    net.learningRate = 0.01
    net.momentum = 0.95
    net.output = "softmax"
    return net


# calculate probit function
def probit(miu, std):
    p = (miu / (1 + np.pi * (std ** 2) / 8) ** 0.5)
    return p


# calculate recursive mean and standard deviation
def meanstditer(miu_old, var_old, x, k):
    miu = miu_old + (x - miu_old) / k
    var = var_old + (x - miu_old) * (x - miu)
    std = np.sqrt(var / k)

    if (len(miu.shape) != 2):
        # print('len !=2')
        # print([np.array([miu]),np.array([std]),np.array([var])])
        return [np.array([miu]), np.array([std]), np.array([var])]
    else:
        return [miu, std, var]


def netffhl(nn, x):
    n = nn.n - 1
    m = x.shape[0]
    ones = np.ones(shape=(m, 1))
    x = np.append(ones, x, axis=1)  # append columns of 1 in matrix for bias
    nn.a[1] = x
    # feedforward from input layer through all the hidden layer
    last = 0
    for i in range(2, n):
        last = i
        if (nn.activation_func == 'sigm'):
            nn.a[i] = sigmoid_array(np.matmul(nn.a[i - 1], nn.W[i - 1].T))
        elif (nn.activation_func == 'relu'):
            nn.a[i] = relu_array(np.matmul(nn.a[i - 1], nn.W[i - 1].T))
        ones = np.ones(shape=(m, 1))
        nn.a[i] = np.append(ones, nn.a[i], axis=1)
    y = nn.a[last]
    return y


def netffsingle(nn, x, y):
    x = np.array([x])
    y = np.array([y])
    n = nn.n
    m = x.shape[0]
    nn.a[1] = x

    # feedforward form input layer through all the hidden layer
    for i in range(2, n):
        if (nn.activation_func == 'sigm'):
            nn.a[i] = sigmoid_array(np.matmul(nn.a[i - 1], nn.W[i - 1].T))
        elif (nn.activation_func == 'relu'):
            nn.a[i] = relu_array(np.matmul(nn.a[i - 1], nn.W[i - 1].T))
        ones = np.ones(shape=(m, 1))
        nn.a[i] = np.append(ones, nn.a[i], axis=1)

    # propagate to the output layer
    if (nn.output == 'sigm'):
        nn.a[n] = sigmoid_array(np.matmul(nn.a[n - 1], nn.W[n - 1].T))
    elif (nn.output == 'linear'):
        nn.a[n] = np.matmul(nn.a[n - 1], nn.W[n - 1].T)
    elif (nn.output == 'softmax'):
        nn.a[n] = softmax_array(np.matmul(nn.a[n - 1], nn.W[n - 1].T))

        # calculate error
    nn.e[1] = y - nn.a[n]
    return nn


#calculate backpropagation
def netbackpropagation(nn):
    n = nn.n
    d = {}
    if(nn.output == 'sigm'):
        d[n] = - nn.e[1] * (nn.a[n] * (1 - nn.a[n]))
    elif(nn.output == 'linear' or nn.output == 'softmax'):
        d[n] = -1 * nn.e[1]
    for i in range(n-1, 1, -1):
        if(nn.activation_func == 'sigm'):
            d_act = nn.a[i] * (1 - nn.a[i])
        elif(nn.activation_func == 'tanh_opt'):
            d_act = 1.7159 * 2/3 * (1 - 1 / (1.7159**2) * (nn.a[i]**2))
        elif(nn.activation_func == 'relu'):
            d_act = np.zeros((1,len(nn.a[i])))
            for i in range(len(d_act)):
                if(nn.a[i + 1]>0):
                    d_act[i] = 0
        if(i+1 == n):
            d[i] = (np.matmul(d[i+1], nn.W[i]) * d_act)
        else:
            d[i] = (np.matmul(d[i+1][1:], nn.W[i]) * d_act)

    for i in range(1, n):
        if(i+1 == n):
            nn.dW[i] = np.matmul(d[i+1].T, nn.a[i])
        else:
            nn.dW[i] = np.matmul(d[i+1][:,1:].T, nn.a[i])
    return nn


#update the weight
def netupdate(nn):
    for i in range(1, nn.n):
        dW = nn.dW[i]
        dW = nn.learningRate * dW
        if(nn.momentum > 0):
            nn.vW[i] = nn.momentum * nn.vW[i] + dW
            dW = nn.vW[i]
        nn.W[i] = nn.W[i] - dW
    return nn


def nettrainsingle(parameter, x, y):
    bb = parameter.nn.W[parameter.nn.hl].shape[1]
    grow = 0
    prune = 0

    # initialize performance matrix
    ly = parameter.nn.hl
    kp = parameter.ev[1]['kp']
    miu_x_old = parameter.ev[1]['miu_x_old']
    var_x_old = parameter.ev[1]['var_x_old']
    kl = parameter.ev[ly]['kl']
    K = parameter.ev[ly]['K']
    node = parameter.ev[ly]['node']
    BIAS2 = parameter.ev[ly]['BIAS2']
    VAR = parameter.ev[ly]['VAR']
    miu_NS_old = parameter.ev[ly]['miu_NS_old']
    var_NS_old = parameter.ev[ly]['var_NS_old']
    miu_NHS_old = parameter.ev[ly]['miu_NHS_old']
    var_NHS_old = parameter.ev[ly]['var_NHS_old']
    miumin_NS = parameter.ev[ly]['miumin_NS']
    miumin_NHS = parameter.ev[ly]['miumin_NHS']
    stdmin_NS = parameter.ev[ly]['stdmin_NS']
    stdmin_NHS = parameter.ev[ly]['stdmin_NHS']

    net = netconfigtrain([1, 1, 1])
    # substitute the weight to be trained to training model
    net.activation_func = parameter.nn.activation_func
    net.W[1] = parameter.nn.W[ly]
    net.vW[1] = parameter.nn.vW[ly]
    net.dW[1] = parameter.nn.dW[ly]
    net.W[2] = parameter.nn.Ws[ly]
    net.vW[2] = parameter.nn.vWs[ly]
    net.dW[2] = parameter.nn.dWs[ly]

    # load data in shuffled
    (N, I) = x.shape
    kk = np.random.permutation(N)
    ones = np.ones(shape=(N, 1))
    x = x[kk]
    x = np.append(ones, x, axis=1)
    y = y[kk]

    # xavier initialization
    n_in = parameter.ev[ly - 1]['K']
    miuNS = np.zeros((N, 1))
    miuminNS = np.zeros((N, 1))
    miuNHS = np.zeros((N, 1))
    miuminNHS = np.zeros((N, 1))

    # main loop, train the model
    for k in range(1, N + 1):
        kp = kp + 1
        kl = kl + 1
        # incremental calculation of x_tail mean and variance
        if (k <= parameter.nn.a[1].shape[0]):
            [miu_x, std_x, var_x] = meanstditer(miu_x_old, var_x_old, parameter.nn.a[1][k - 1], kp)
            miu_x_old = miu_x
            var_x_old = var_x

            # expectation of z
            py = probit(miu_x, std_x).T

            for ii in range(1, parameter.nn.hl + 1):
                if (ii == parameter.nn.hl):
                    py = sigmoid_array(np.matmul(net.W[1], py))
                else:
                    py = sigmoid_array(np.matmul(parameter.nn.W[ii], py))
                py = np.append(np.array([[1]]), py, axis=0)
                if (ii == 1):
                    Ey2 = py ** 2
            Ey = py
            Ez = np.matmul(net.W[2], Ey)
            Ez = softmax_array(Ez.T).T

            if (parameter.nn.hl > 1):
                py = Ey2
                for ii in range(2, parameter.nn.hl + 1):
                    if (ii == parameter.nn.hl):
                        py = sigmoid_array(np.matmul(net.W[1], py))
                    else:
                        py = sigmoid_array(np.matmul(parameter.nn.W[ii], py))
                    py = np.append(np.array([[1]]), py, axis=0)
                Ey2 = py
            Ez2 = np.matmul(net.W[2], Ey2)
            Ez2 = softmax_array(Ez2.T).T

            # Network mean calculation
            bias2 = (Ez - np.array([[1, 0]]).T) ** 2
            ns = bias2
            NS = np.linalg.norm(ns, 'fro')

            # Incremental calculation of NS mean and variance
            [miu_NS, std_NS, var_NS] = meanstditer(miu_NS_old, var_NS_old, NS, kp)
            miu_NS_old = miu_NS
            var_NS_old = var_NS
            miustd_NS = miu_NS + std_NS
            miuNS[k - 1] = miu_NS

            if (kl <= 1 or grow == 1):
                miumin_NS = miu_NS
                stdmin_NS = std_NS
            else:
                if (miu_NS < miumin_NS):
                    miumin_NS = miu_NS
                    stdmin_NS = std_NS
                if (std_NS < stdmin_NS):
                    stdmin_NS = std_NS
            miuminNS = miumin_NS
            miustdmin_NS = miumin_NS + (1.3 * np.exp(-NS) + 0.7) * stdmin_NS
            BIAS2[kl] = miu_NS

            # growing hidden unit
            if (miustd_NS >= miustdmin_NS and kl > 1):
                grow = 1
                K = K + 1
                print('The new node no {} is FORMED around sample {}'.format(K, k))
                node[kp] = K
                W_app = np.random.normal(0, np.sqrt(2 / (n_in + 1)), size=(1, bb))
                net.W[1] = np.append(net.W[1], W_app, axis=0)
                net.vW[1] = np.append(net.vW[1], np.zeros((1, bb)), axis=0)
                net.dW[1] = np.append(net.dW[1], np.zeros((1, bb)), axis=0)

                W2_app = np.random.normal(0, np.sqrt(2 / (K + 1)), size=(parameter.nn.size[-1], 1))
                net.W[2] = np.append(net.W[2], W2_app, axis=1)
                net.vW[2] = np.append(net.vW[2], np.zeros((parameter.nn.size[-1], 1)), axis=1)
                net.dW[2] = np.append(net.dW[2], np.zeros((parameter.nn.size[-1], 1)), axis=1)
            else:
                grow = 0
                node[kp] = K

            # Network variance calculation
            var = Ez2 - Ez ** 2
            NHS = np.linalg.norm(var, 'fro')

            # Incremental calculation of NHS mean and variance
            [miu_NHS, std_NHS, var_NHS] = meanstditer(miu_NHS_old, var_NHS_old, NHS, kp)
            miu_NHS_old = miu_NHS
            var_NHS_old = var_NHS
            miustd_NHS = miu_NHS + std_NHS
            miuNHS[k - 1] = miu_NHS

            if (kl <= I + 1 or prune == 1):
                miumin_NHS = miu_NHS
                stdmin_NHS = std_NHS
            else:
                if (miu_NHS < miumin_NHS):
                    miumin_NHS = miu_NHS
                if (std_NHS < stdmin_NHS):
                    stdmin_NHS = std_NHS
            miustdmin_NHS = miumin_NHS + (2.6 * np.exp(-NHS) + 1.4) * stdmin_NHS
            VAR[kl] = miu_NHS

            # Pruning hidden unit
            if (grow == 0 and K > 1 and miustd_NHS >= miustdmin_NHS and kl > I + 1):
                HS = Ey[1:]
                BB = np.argmin(HS)
                print('The node no {} is PRUNED around sample {}'.format(BB + 1, k))
                prune = 1
                K = K - 1
                node[kp] = K
                net.W[1] = np.delete(net.W[1], BB, axis=0)
                net.vW[1] = np.delete(net.vW[1], BB, axis=0)
                net.dW[1] = np.delete(net.dW[1], BB, axis=0)

                net.W[2] = np.delete(net.W[2], BB + 1, axis=1)
                net.vW[2] = np.delete(net.vW[2], BB + 1, axis=1)
                net.dW[2] = np.delete(net.dW[2], BB + 1, axis=1)
            else:
                prune = 0
                node[kp] = K

            # feedforward
            net = netffsingle(net, x[k - 1], y[k - 1])

        # feedforward #2, executed if there is a hidden node changing
        net = netbackpropagation(net)
        net = netupdate(net)

    # subsitutte weight back to main model
    parameter.nn.W[ly] = net.W[1]
    parameter.nn.Ws[ly] = net.W[2]

    # reset momentum and grandient
    parameter.nn.vW[ly] = net.vW[1] * 0
    parameter.nn.dW[ly] = net.dW[1] * 0
    parameter.nn.vWs[ly] = net.vW[2] * 0
    parameter.nn.dWs[ly] = net.dW[2] * 0

    # substitute the recursive calculation
    parameter.ev[ly]['kl'] = kl
    parameter.ev[ly]['K'] = K
    parameter.ev[ly]['node'] = node
    parameter.ev[ly]['BIAS2'] = BIAS2
    parameter.ev[ly]['VAR'] = VAR
    parameter.ev[ly]['miu_NS_old'] = miu_NS_old
    parameter.ev[ly]['var_NS_old'] = var_NS_old
    parameter.ev[ly]['miu_NHS_old'] = miu_NHS_old
    parameter.ev[ly]['var_NHS_old'] = var_NHS_old
    parameter.ev[ly]['miumin_NS'] = miumin_NS
    parameter.ev[ly]['miumin_NHS'] = miumin_NHS
    parameter.ev[ly]['stdmin_NS'] = stdmin_NS
    parameter.ev[ly]['stdmin_NHS'] = stdmin_NHS
    return parameter


def nettrainparallel(parameter, y):
    bb = parameter.nn.W[parameter.nn.index].shape[1]
    grow = 0
    prune = 0
    # initialize performance matrix
    ly = parameter.nn.index
    kp = parameter.ev[1]['kp']
    miu_x_old = parameter.ev[1]['miu_x_old']
    var_x_old = parameter.ev[1]['var_x_old']
    kl = parameter.ev[ly]['kl']
    K = parameter.ev[ly]['K']
    node = parameter.ev[ly]['node']
    BIAS2 = parameter.ev[ly]['BIAS2']
    VAR = parameter.ev[ly]['VAR']
    miu_NS_old = parameter.ev[ly]['miu_NS_old']
    var_NS_old = parameter.ev[ly]['var_NS_old']
    miu_NHS_old = parameter.ev[ly]['miu_NHS_old']
    var_NHS_old = parameter.ev[ly]['var_NHS_old']
    miumin_NS = parameter.ev[ly]['miumin_NS']
    miumin_NHS = parameter.ev[ly]['miumin_NHS']
    stdmin_NS = parameter.ev[ly]['stdmin_NS']
    stdmin_NHS = parameter.ev[ly]['stdmin_NHS']

    # initiate training model
    net = netconfigtrain([1, 1, 1])
    # substitute the weight to be trained to training model
    net.activation_func = parameter.nn.activation_func
    net.W[1] = parameter.nn.W[ly]
    net.vW[1] = parameter.nn.vW[ly]
    net.dW[1] = parameter.nn.dW[ly]
    net.W[2] = parameter.nn.Ws[ly]
    net.vW[2] = parameter.nn.vWs[ly]
    net.dW[2] = parameter.nn.dWs[ly]

    # load data in shuffled
    x = parameter.nn.a[ly]
    (N, I) = x.shape
    kk = np.random.permutation(N)
    x = x[kk]
    y = y[kk]

    # xavier initialization
    if (ly > 1):
        n_in = parameter.ev[ly - 1]['K']
    else:
        n_in = parameter.nn.size[0]

    miuNS = np.zeros((N, 1))
    miuminNS = np.zeros((N, 1))
    miuNHS = np.zeros((N, 1))
    miuminNHS = np.zeros((N, 1))

    # main loop, train the model
    for k in range(1, N + 1):
        kp = kp + 1
        kl = kl + 1
        # incremental calculation of x_tail mean and variance
        [miu_x, std_x, var_x] = meanstditer(miu_x_old, var_x_old, parameter.nn.a[1][k - 1], kp)
        miu_x_old = miu_x
        var_x_old = var_x

        # expectation of z
        py = probit(miu_x, std_x).T

        for ii in range(1, parameter.nn.index + 1):
            if (ii == parameter.nn.index):
                py = sigmoid_array(np.matmul(net.W[1], py))
            else:
                py = sigmoid_array(np.matmul(parameter.nn.W[ii], py))
            py = np.append(np.array([[1]]), py, axis=0)
            if (ii == 1):
                Ey2 = py ** 2

        Ey = py
        Ez = np.matmul(net.W[2], Ey)
        Ez = softmax_array(Ez.T).T

        if (parameter.nn.hl > 1):
            py = Ey2
            for ii in range(2, parameter.nn.index + 1):
                if (ii == parameter.nn.index):
                    py = sigmoid_array(np.matmul(net.W[1], py))
                else:
                    py = sigmoid_array(np.matmul(parameter.nn.W[ii], py))
                py = np.append(np.array([[1]]), py, axis=0)
            Ey2 = py
        Ez2 = np.matmul(net.W[2], Ey2)
        Ez2 = softmax_array(Ez2.T).T

        # Network mean calculation
        bias2 = (Ez - np.array([[1, 0]]).T) ** 2
        ns = bias2
        NS = np.linalg.norm(ns, 'fro')

        # Incremental calculation of NS mean and variance
        [miu_NS, std_NS, var_NS] = meanstditer(miu_NS_old, var_NS_old, NS, kp)
        miu_NS_old = miu_NS
        var_NS_old = var_NS
        miustd_NS = miu_NS + std_NS
        miuNS[k - 1] = miu_NS

        if (kl <= 1 or grow == 1):
            miumin_NS = miu_NS
            stdmin_NS = std_NS
        else:
            if (miu_NS < miumin_NS):
                miumin_NS = miu_NS
                stdmin_NS = std_NS
            if (std_NS < stdmin_NS):
                stdmin_NS = std_NS
        miuminNS[k - 1] = miumin_NS
        miustdmin_NS = miumin_NS + (1.3 * np.exp(-NS) + 0.7) * stdmin_NS
        BIAS2[kl] = miu_NS

        # growing hidden unit
        if (miustd_NS >= miustdmin_NS and kl > 1):
            grow = 1
            K = K + 1
            print('The new node no {} is FORMED around sample {}'.format(K, k))
            node[kp] = K
            W_app = np.random.normal(0, np.sqrt(2 / (n_in + 1)), size=(1, bb))
            net.W[1] = np.append(net.W[1], W_app, axis=0)
            net.vW[1] = np.append(net.vW[1], np.zeros((1, bb)), axis=0)
            net.dW[1] = np.append(net.dW[1], np.zeros((1, bb)), axis=0)

            W2_app = np.random.normal(0, np.sqrt(2 / (K + 1)), size=(parameter.nn.size[-1], 1))
            net.W[2] = np.append(net.W[2], W2_app, axis=1)
            net.vW[2] = np.append(net.vW[2], np.zeros((parameter.nn.size[-1], 1)), axis=1)
            net.dW[2] = np.append(net.dW[2], np.zeros((parameter.nn.size[-1], 1)), axis=1)

            if (ly < parameter.nn.hl):
                wNext = parameter.nn.W[ly + 1].shape[0]
                parameter.nn.W[ly + 1] = np.append(parameter.nn.W[ly + 1],
                                                   np.random.normal(0, np.sqrt(2 / (K + 1)), size=(wNext, 1)), axis=1)
                parameter.nn.vW[ly + 1] = np.append(parameter.nn.vW[ly + 1], np.zeros((wNext, 1)), axis=1)
                parameter.nn.dW[ly + 1] = np.append(parameter.nn.dW[ly + 1], np.zeros((wNext, 1)), axis=1)
        else:
            grow = 0
            node[kp] = K
        # Network variance calculation
        #         ho = np.array([[0.7498],[0.2502]])
        #         he = np.array([[0.7856],[0.2144]])
        #         uy = ho-he**2
        #         ey = np.linalg.norm(uy, 'fro')

        var = Ez2 - Ez ** 2
        NHS = np.linalg.norm(var, 'fro')

        # Incremental calculation of NHS mean and variance
        [miu_NHS, std_NHS, var_NHS] = meanstditer(miu_NHS_old, var_NHS_old, NHS, kp)
        miu_NHS_old = miu_NHS
        var_NHS_old = var_NHS
        miustd_NHS = miu_NHS + std_NHS
        miuNHS[k - 1] = miu_NHS

        if (kl <= I + 1 or prune == 1):
            miumin_NHS = miu_NHS
            stdmin_NHS = std_NHS
        else:
            if (miu_NHS < miumin_NHS):
                miumin_NHS = miu_NHS
            if (std_NHS < stdmin_NHS):
                stdmin_NHS = std_NHS
        miuminNHS[k - 1] = miumin_NHS
        miustdmin_NHS = miumin_NHS + (2.6 * np.exp(-NHS) + 1.4) * stdmin_NHS
        VAR[kl] = miu_NHS

        # Pruning hidden unit
        if (grow == 0 and K > 1 and miustd_NHS >= miustdmin_NHS and kl > I + 1):
            HS = Ey[1:]
            BB = np.argmin(HS)
            print('The node no {} is PRUNED around sample {}'.format(BB + 1, k))
            prune = 1
            K = K - 1
            node[kp] = K
            net.W[1] = np.delete(net.W[1], BB, axis=0)
            net.vW[1] = np.delete(net.vW[1], BB, axis=0)
            net.dW[1] = np.delete(net.dW[1], BB, axis=0)

            net.W[2] = np.delete(net.W[2], BB + 1, axis=1)
            net.vW[2] = np.delete(net.vW[2], BB + 1, axis=1)
            net.dW[2] = np.delete(net.dW[2], BB + 1, axis=1)
            if (ly < parameter.nn.hl):
                parameter.nn.W[ly + 1] = np.delete(parameter.nn.W[ly + 1], BB + 1, axis=1)
                parameter.nn.vW[ly + 1] = np.delete(parameter.nn.vW[ly + 1], BB + 1, axis=1)
                parameter.nn.dW[ly + 1] = np.delete(parameter.nn.dW[ly + 1], BB + 1, axis=1)
        else:
            prune = 0
            node[kp] = K
        # feedforward
        net = netffsingle(net, x[k - 1], y[k - 1])

        # feedforward #2, executed if there is a hidden node changing
        net = netbackpropagation(net)
        net = netupdate(net)

    # subsitutte weight back to main model
    parameter.nn.W[ly] = net.W[1]
    parameter.nn.Ws[ly] = net.W[2]

    # reset momentum and grandient
    parameter.nn.vW[ly] = net.vW[1] * 0
    parameter.nn.dW[ly] = net.dW[1] * 0
    parameter.nn.vWs[ly] = net.vW[2] * 0
    parameter.nn.dWs[ly] = net.dW[2] * 0

    # substitute the recursive calculation
    parameter.ev[1]['kp'] = kp
    parameter.ev[1]['miu_x_old'] = miu_x_old
    parameter.ev[1]['var_x_old'] = var_x_old
    parameter.ev[ly]['kl'] = kl
    parameter.ev[ly]['K'] = K
    parameter.ev[ly]['node'] = node
    parameter.ev[ly]['BIAS2'] = BIAS2
    parameter.ev[ly]['VAR'] = VAR
    parameter.ev[ly]['miu_NS_old'] = miu_NS_old
    parameter.ev[ly]['var_NS_old'] = var_NS_old
    parameter.ev[ly]['miu_NHS_old'] = miu_NHS_old
    parameter.ev[ly]['var_NHS_old'] = var_NHS_old
    parameter.ev[ly]['miumin_NS'] = miumin_NS
    parameter.ev[ly]['miumin_NHS'] = miumin_NHS
    parameter.ev[ly]['stdmin_NS'] = stdmin_NS
    parameter.ev[ly]['stdmin_NHS'] = stdmin_NHS
    # print(BIAS2)
    return parameter


# method for precission, recall, f1 score and gmean
def performance_summary(act_raw, out_raw, nclass):
    Act = to_one_hot(act_raw, nclass)
    Out = to_one_hot(out_raw, nclass)
    recall = compute_recall(Act, Out, nclass)
    precission = compute_precission(Act, Out, nclass)
    f_measure = compute_f_measure(Act, Out, nclass)
    g_mean = compute_g_mean(recall, nclass)
    return [g_mean, f_measure, precission, recall]


def compute_g_mean(recall, nclass):
    g_mean = (np.prod(recall)) ** (1 / nclass)
    return g_mean


def compute_f_measure(Act, Out, nclass):
    f_measure = np.zeros((1, nclass))
    for c in range(nclass):
        f_measure[0][c] = (2 * (Act[:, c].T.dot(Out[:, c]))) / (np.sum(Out[:, c]) + np.sum(Act[:, c]))
    f_measure[np.isnan(f_measure)] = 1
    return f_measure


def compute_precission(Act, Out, nclass):
    precission = np.zeros((1, nclass))
    for c in range(nclass):
        precission[0][c] = (Act[:, c].T.dot(Out[:, c])) / np.sum(Out[:, c])
    precission[np.isnan(precission)] = 1
    return precission


def compute_recall(Act, Out, nclass):
    recall = np.zeros((1, nclass))
    for c in range(nclass):
        recall[0][c] = (Act[:, c].T.dot(Out[:, c])) / np.sum(Act[:, c])
    recall[np.isnan(recall)] = 1
    return recall


def to_one_hot(x, nclass):
    y = np.zeros((x.shape[0], nclass))
    for i in range(len(x)):
        y[i][x[i]] = 1
    return y

# Notebook code-----------------------------------------------------------------------------------

def pd_one_hot_encode(df, column_names, preserve_columns=False):
    if preserve_columns:
        # http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example
        for column_name in column_names:
            df1 = pd.concat([df, pd.get_dummies(df[column_names], prefix=column_names)], axis=1)
    else:
        # http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example
        df1 = pd.get_dummies(df, columns=column_names, prefix=column_names)
    return df1


# df = pd.read_csv(args.input)
# print(df.describe())
# n_feature = len(df.columns) - 1
# print(n_feature)
# print(df.columns[len(df.columns) - 1])
# df_hot_columns = []
# df_hot_columns.append(df.columns[len(df.columns) - 1])
# df = pd_one_hot_encode(df, column_names=df_hot_columns, preserve_columns=False)
# print(df.describe())
# print(len(df.columns))
# data = df.to_numpy()
# data = np.genfromtxt('spam_corpus_class_map_random_labels_one_hot.csv', delimiter=',', skip_header=1)
# t1 = time.time()
# data_load_time = t1-t0
# print('data_load_time={}s'.format(data_load_time))
# exit(0)
# data1 = scipy.io.loadmat('sea.mat')
# data = data1.get('data')




def test_ADL(data, n_feature):
    t2 = time.time()
    (nData, n_column) = data.shape
    M = n_column - n_feature
    preq_data = data[:,0:n_feature]
    preq_label = data[:,n_feature:]
    chunk_size = args.batchsize
    no_of_chunk = int(nData/chunk_size)

    drift = {}
    HL = {}
    buffer_x = []
    buffer_T = []
    tTest = []
    tTarget = []
    act = []
    out = []
    #initiate model
    K = 1 #initial node
    network = nn([n_feature, K, M])

    #initiate node evolving iterative parameters
    layer = 1 #initial layer
    parameter = Parameter(network, layer,K)
    performance = Performance()

    # initiate drift detection parameter
    alpha_w = 0.0005
    alpha_d = 0.0001
    alpha   = 0.0001

    #initiate layer merging iterative parameters
    covariance = np.zeros((1,1,2))
    covariance_old             = covariance
    threshold                  = 0.05

    ClassificationRate = {}
    for count in range(0,no_of_chunk):
        # prepare data
        n = count + 1
        minibatch_data  = preq_data [(n-1)*chunk_size:n*chunk_size]
        minibatch_label = preq_label[(n-1)*chunk_size:n*chunk_size]

        # neural network testing
        print('Chunk: {} of {}'.format(n, no_of_chunk))
        print('Discriminative Testing: running ...')
        parameter.nn.t = n
        [parameter.nn] = nettestparallel(parameter.nn,minibatch_data,minibatch_label,parameter.ev)

        #metrics calculation
        parameter.Loss[n] = parameter.nn.L[parameter.nn.index]
        if(n == 1):
            tTest = parameter.nn.sigma.copy()
            act = parameter.nn.act.copy()
            out = parameter.nn.out.copy()
            parameter.residual_error = np.append(out,parameter.nn.residual_error,axis=0)
        else:
            tTest = np.append(tTest,parameter.nn.sigma,axis=0)
            act = np.append(act,parameter.nn.act,axis=0)
            out = np.append(out,parameter.nn.out,axis=0)
            parameter.residual_error = np.append(out,parameter.nn.residual_error,axis=0)
        parameter.cr[n] = parameter.nn.cr
        ClassificationRate[n] = np.array(list(parameter.cr.values())).mean()
        print('Classification rate {}'.format(ClassificationRate[n]))
        print('Discriminative Testing: ... finished')

        #statistical measure
        performance.ev[n] = {}
        [performance.ev[n]['f_measure'], performance.ev[n]['g_mean'] ,performance.ev[n]['recall'],performance.ev[n]['precision']] = performance_summary(parameter.nn.act, parameter.nn.out, M)
        #last chunk only for testing process
        if(n == no_of_chunk):
            print('=========Parallel Autonomous Deep Learning is finished=========')
            break

        #Drift detection: output space
        if(n>1):
            cuttingpoint = 0
            pp = minibatch_label.shape[0]
            F_cut = np.zeros((pp,1))
            F_cut[parameter.nn.bad] = 1
            Fupper = np.max(F_cut)
            Flower = np.min(F_cut)
            miu_F = np.mean(F_cut)

            for idx in range(pp):
                cut = idx + 1
                miu_G = np.mean(F_cut[0:cut])
                Gupper = np.max(F_cut[0:cut])
                Glower = np.min(F_cut[0:cut])
                epsilon_G = (Gupper - Glower) * np.sqrt(((pp)/(2*cut*(pp)) * np.log(1/alpha)))
                epsilon_F = (Fupper - Flower) * np.sqrt(((pp)/(2*cut*(pp)) * np.log(1/alpha)))
                if ((epsilon_G + miu_G) >= (miu_F + epsilon_F) and cut<pp):
                    cuttingpoint = cut
                    miu_H = np.mean(F_cut[(cuttingpoint):])
                    epsilon_D = (Fupper - Flower) * np.sqrt(((pp-cuttingpoint)/(2*cuttingpoint*(pp-cuttingpoint)) * np.log(1/alpha_d)))
                    epsilon_W = (Fupper - Flower) * np.sqrt(((pp-cuttingpoint)/(2*cuttingpoint*(pp-cuttingpoint)) * np.log(1/alpha_w)))
                    break
            if(cuttingpoint == 0):
                miu_H = miu_F
                epsilon_D = (Fupper - Flower) * np.sqrt(((pp)/(2*cut*(pp)) * np.log(1/alpha_d)))
                epsilon_W = (Fupper - Flower) * np.sqrt(((pp)/(2*cut*(pp)) * np.log(1/alpha_w)))

            #DRIFT STATUS
            if((np.abs(miu_G - miu_H)) > epsilon_D and cuttingpoint>1):
                st = 1
                print('Drift state: DRIFT')
                layer = layer+1
                parameter.nn.n = parameter.nn.n + 1
                parameter.nn.hl = layer
                print('The new Layer no {} is FORMED around chunk {}'.format(layer, n))

                #Initiate NN weight parameters
                ii = parameter.nn.W[layer-1].shape[0]
                parameter.nn.W[layer] = np.random.normal(0,np.sqrt(2/(ii+1)),size = (1, (ii+1)))
                parameter.nn.vW[layer] = np.zeros((1,ii+1))
                parameter.nn.dW[layer] = np.zeros((1,ii+1))

                #Initiate new classifier weight
                parameter.nn.Ws[layer]  = np.random.normal(0,1,size = (M,2))
                parameter.nn.vWs[layer] = np.zeros((M,2))
                parameter.nn.dWs[layer] = np.zeros((M,2))

                #Initiate new voting weight
                parameter.nn.beta[layer] = 1
                parameter.nn.betaOld[layer] = 1
                parameter.nn.p[layer] = 1

                # Initiate iterative parameters
                parameter.ev[layer] = {}
                parameter.ev[layer]['layer ']      = layer
                parameter.ev[layer]['kl']          = 0
                parameter.ev[layer]['K']           = 1
                parameter.ev[layer]['cr']           = 0
                parameter.ev[layer]['node']        = {}
                parameter.ev[layer]['miu_NS_old']  = 0
                parameter.ev[layer]['var_NS_old']  = 0
                parameter.ev[layer]['miu_NHS_old'] = 0
                parameter.ev[layer]['var_NHS_old'] = 0
                parameter.ev[layer]['miumin_NS']   = []
                parameter.ev[layer]['miumin_NHS']  = []
                parameter.ev[layer]['stdmin_NS']   = []
                parameter.ev[layer]['stdmin_NHS']  = []
                parameter.ev[layer]['BIAS2']       = {}
                parameter.ev[layer]['VAR']         = {}

                #check buffer
                if(len(buffer_x) == 0):
                    h = parameter.nn.a[len(parameter.nn.a)][:,1:]
                    z = minibatch_label
                else:
                    buffer_x = netffhl(parameter.nn, buffer_x)
                    h = np.append(buffer_x[:,1:],parameter.nn.a[len(parameter.nn.a)][:,1:],axis=0)
                    if(len(buffer_T) == 0):
                        z = np.append(buffer_T,minibatch_label ,axis=0)
                    else:
                        z = minibatch_label

                #Discriminative training for new layer
                print('Discriminative Training for new layer: running ...')
                parameter = nettrainsingle(parameter,h,z)
                print('Discriminative Training for new layer: ... finished')
                buffer_x = []
                buffer_T = []
            elif((np.abs(miu_G - miu_H)) >= epsilon_W and (np.abs(miu_G - miu_H)) < epsilon_D):
                st = 2
                print('Drift state: WARNING')
                buffer_x = minibatch_data
                buffer_T = minibatch_label
            else:
                st = 3
                print('Drift state: STABLE')
                buffer_x = []
                buffer_T = []
        else:
            st = 3
            print('Drift state: STABLE')
            buffer_x = []
            buffer_T = []
        drift[n] = st
        HL[n] = checkbeta(parameter.nn.beta)
        parameter.wl[n] = parameter.nn.index

        #Discriminative training for winning layer
        if(st != 1):
            print('Discriminative Training: running ...')
            parameter = nettrainparallel(parameter, minibatch_label)
            print('Discriminative Training: ... finished')

        #Clear current data chunk
        parameter.nn.a = {}
        print('=========Hidden layer number {} was updated========='.format(parameter.nn.index))

        t3 = time.time()
        test_and_train_time = t3 - t2
        # file1.write("id,mean_acc_[M0],current_acc_[M0],mean_kappa_[M0],current_kappa_[M0],training_time_[M0],testing_time_[M0],total_running_time_[M0],model_size_[M0]\n")
        file1.write("{},{},{},0.0,0.0,0.0,0.0,{},0.0\n".format(n*chunk_size, ClassificationRate[n], parameter.nn.cr, test_and_train_time))
    return parameter,performance


# if __name__ == "__main__":
#     parameter, performance = test_ADL(data, n_feature)
#     file1.close()


class AutonomousNN(BaseSKMObject, ClassifierMixin):
    def __init__(self, n_features, n_classes):
        self.drift = {}
        self.HL = {}
        self.buffer_x = []
        self.buffer_T = []
        self.tTest = []
        self.tTarget = []
        self.act = []
        self.out = []
        # initiate model
        self.K = 1  # initial node
        self.M = n_classes
        self.network = nn([n_features, self.K, self.M])

        # initiate node evolving iterative parameters
        self.layer = 1  # initial layer
        self.parameter = Parameter(self.network, self.layer, self.K)
        self.performance = Performance()

        # initiate drift detection parameter
        self.alpha_w = 0.0005
        self.alpha_d = 0.0001
        self.alpha = 0.0001

        # initiate layer merging iterative parameters
        self.covariance = np.zeros((1, 1, 2))
        self.covariance_old = self.covariance
        self.threshold = 0.05

        self.ClassificationRate = {}

        super().__init__()

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        r, c = get_dimensions(X)
        #
        # if not self.network_initialized:
        #     if len(self.network_config) > 0:
        #         self.network_config[0]['input_d'] = c
        #     else:
        #         self.network_config = auto_configure_layers(input_dimensions=c, return_basic=True)
        #     self.initialize_network(network_layers=self.network_config)
        #
        # for i in range(r):
        #     self.forward_prop(X[i])
        #     self.backward_prop(self.label_to_class[y[i]])
        return self

    def predict(self, X, y):
        y_proba = self.predict_proba(X, y)
        y_pred = np.argmax(y_proba, axis=1)
        return vectorized_map_class_to_label(y_pred, class_to_label_map=self.class_to_label)

    def predict_proba(self, X, y):
        proba = []
        r, c = get_dimensions(X)
        #
        # if not self.network_initialized:
        #     if len(self.network_config) > 0:
        #         self.network_config[0]['input_d'] = c
        #     else:
        #         self.network_config = auto_configure_layers(input_dimensions=c, return_basic=True)
        #     self.initialize_network(network_layers=self.network_config)
        #
        # for i in range(r):
        #     self.forward_prop(X[i])
        #     proba.append([1 - self.layer[len(self.layer) - 1]['a'], self.layer[len(self.layer) - 1]['a']])

        # prepare data
        # n = count + 1
        # minibatch_data = preq_data[(n - 1) * chunk_size:n * chunk_size]
        minibatch_data = X
        # minibatch_label = preq_label[(n - 1) * chunk_size:n * chunk_size]
        minibatch_label = to_one_hot(y, self.M)

        # neural network testing
        # print('Chunk: {} of {}'.format(n, no_of_chunk))
        # print('Discriminative Testing: running ...')
        # parameter.nn.t = n
        self.parameter.nn.t += 1
        n = self.parameter.nn.t
        print('Chunk: {}'.format(self.parameter.nn.t))
        print('Discriminative Testing: running ...')

        [self.parameter.nn] = nettestparallel(self.parameter.nn, minibatch_data, minibatch_label, self.parameter.ev)

        # metrics calculation
        self.parameter.Loss[n] = self.parameter.nn.L[parameter.nn.index]
        if (n == 1):
            self.tTest = self.parameter.nn.sigma.copy()
            self.act = self.parameter.nn.act.copy()
            self.out = self.parameter.nn.out.copy()
            self.parameter.residual_error = np.append(out, self.parameter.nn.residual_error, axis=0)
        else:
            self.tTest = np.append(self.tTest, self.parameter.nn.sigma, axis=0)
            self.act = np.append(self.act, self.parameter.nn.act, axis=0)
            self.out = np.append(self.out, self.parameter.nn.out, axis=0)
            self.parameter.residual_error = np.append(self.out, self.parameter.nn.residual_error, axis=0)
        self.parameter.cr[n] = self.parameter.nn.cr
        self.ClassificationRate[n] = np.array(list(self.parameter.cr.values())).mean()
        print('Classification rate {}'.format(self.ClassificationRate[n]))
        print('Discriminative Testing: ... finished')

        return np.asarray(proba)

