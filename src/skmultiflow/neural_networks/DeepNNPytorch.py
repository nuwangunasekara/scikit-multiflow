import numpy as np
from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils.utils import *
from skmultiflow.neural_networks.utils import *

import torch
import torch.nn as nn
import torch.optim as optim

default_network_layers = [{'neurons': 0, 'input_d': 0}, {'neurons': 2 ** 8, 'g': 3}, {'neurons': 1, 'g': 1}]

class PyNet(nn.Module):
    def __init__(self, nn_layers=None):
        super(PyNet, self).__init__()
        if nn_layers is None:
            return
        linear = []
        self.f = []
        for l in range(1, len(nn_layers), 1):
            if l == 1:
                linear.append(nn.Linear(nn_layers[0]['input_d'], nn_layers[l]['neurons']))
            else:
                linear.append(nn.Linear(nn_layers[l - 1]['neurons'], nn_layers[l]['neurons']))
            if nn_layers[l]['g'] == Tanh:
                self.f.append(nn.Tanh())
            elif nn_layers[l]['g'] == Sigmoid:
                self.f.append(nn.Sigmoid())
            elif nn_layers[l]['g'] == Relu:
                self.f.append(nn.ReLU())
            elif nn_layers[l]['g'] == LeakyRelu:
                self.f.append(nn.LeakyReLU())
            else:
                pass
        self.linear = nn.ModuleList(linear)

    def forward(self, x):
        for i, l in enumerate(self.linear):
            x = self.f[i](l(x))
        return x


class DeepNNPytorch(BaseSKMObject, ClassifierMixin):
    def __init__(self,
                 learning_rate=0.03,
                 network_layers=default_network_layers,
                 class_labels=['0','1'],  # {'up':0,'down':1}
                 use_cpu=True):
        # configuration variables (which has the same name as init parameters)
        self.learning_rate = learning_rate
        self.network_layers = network_layers
        self.class_labels = class_labels
        self.use_cpu = use_cpu

        # status variables
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.loss = None
        self.device = None
        self.class_to_label = {}
        self.label_to_class = {}

        self.init_values()

        super().__init__()

    def init_values(self):
        # init status variables
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.loss = None
        self.device = None
        self.class_to_label = {}
        self.label_to_class = {}

        initialize_network = False

        for i in range(len(self.class_labels)):
            self.class_to_label.update({i: self.class_labels[i]})
            self.label_to_class.update({self.class_labels[i]: i})
        # print('class_to_label=', self.class_to_label)
        # print('label_to_class=', self.label_to_class)
        # print('label_to_class=', self.learning_rate)

        if isinstance(self.network_layers, nn.Module):
            self.net = self.network_layers
            self.initialize_net_para()
        elif isinstance(self.network_layers, list):
            if self.network_layers[0]['input_d'] is None or self.network_layers[0]['input_d'] == 0:
                # wait till we receive the first instance to get input dimensions
                # to initialize the passed-in network
                self.network_layers[0]['input_d'] = 0
            else:
                initialize_network = True
        else:
            self.network_layers = default_network_layers
            print('Unknown network type passed in, set the network type to default: {}'.format(self.network_layers))

        if initialize_network:
            self.initialize_network(self.network_layers)

        if self.use_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

    def initialize_net_para(self):
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate)
        # for multi class classification
        # criterion = nn.CrossEntropyLoss()
        # for binary classification
        # combines a Sigmoid layer
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.BCELoss()
        print('Network configuration: \n{}'.format(self))

    def initialize_network(self, network_layers=None):
        self.net = PyNet(network_layers)
        self.initialize_net_para()

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        r, c = get_dimensions(X)

        if self.net is None:
            self.network_layers[0]['input_d'] = c
            self.initialize_network(self.network_layers)

        for i in range(r):
            # self.forward_prop(X[i])
            # self.backward_prop(self.label_to_class[y[i]])
            x = torch.from_numpy(X[i])
            yy = torch.from_numpy(np.array(y[i]))
            x = x.view(1, -1).float()
            yy = yy.view(1, -1).float()
            x.unsqueeze(0)
            yy.unsqueeze(0)
            if torch.cuda.is_available():
                if self.device.type == 'cpu':
                    pass
                else:
                    x = x.to(self.device)
                    yy = yy.to(self.device)
            else:
                pass

            self.optimizer.zero_grad()  # zero the gradient buffers
            # # forward propagation
            output = self.net(x)

            # backward propagation
            # print(self.learning_rate)
            # print(self.net.linear[0].weight.data)
            if self.learning_rate > 0.0:
                # print('here')
                self.loss = self.criterion(output, yy)
                self.loss.backward()
                self.optimizer.step()  # Does the update

        return self

    def predict(self, X):
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
        return vectorized_map_class_to_label(y_pred, class_to_label_map=self.class_to_label)

    def predict_proba(self, X):
        proba = []
        r, c = get_dimensions(X)

        if self.net is None:
            self.network_layers[0]['input_d'] = c
            self.initialize_network(self.network_layers)

        for i in range(r):
            x = torch.from_numpy(X[i])
            x = x.view(1, -1).float()
            x.unsqueeze(0)
            y_prob = self.net(x)
            proba.append([1 - y_prob, y_prob])

        return np.asarray(proba)

    def reset(self):
        # configuration variables (which has the same name as init parameters) should be copied by the caller function
        self.init_values()
        return self
