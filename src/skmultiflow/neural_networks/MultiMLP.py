import sys
import threading

import numpy as np
from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils.utils import *
from skmultiflow.neural_networks.utils import *
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.drift_detection import ADWIN

import torch
import torch.nn as nn
import torch.optim as optim

default_hidden_layers = [{'neurons': 2 ** 10, 'nonlinearity': Relu}]

OP_TYPE_SGD = 'SGD'
OP_TYPE_SGD_NC = 'SGD-NC'
OP_TYPE_ADAGRAD = 'Adagrad'
OP_TYPE_ADAGRAD_NC = 'Adagrad-NC'
OP_TYPE_RMSPROP = 'RMSprop'
OP_TYPE_RMSPROP_NC = 'RMSprop-NC'
OP_TYPE_ADADELTA = 'Adadelta'
OP_TYPE_ADADELTA_NC = 'Adadelta-NC'
OP_TYPE_ADAM = 'Adam'
OP_TYPE_ADAM_NC = 'Adam-NC'
OP_TYPE_ADAM_AMSG = 'Adam-AMSG'
OP_TYPE_ADAM_AMSG_NC = 'Adam-AMSG-NC'


class PyNet(nn.Module):
    def __init__(self, hidden_layers: list = None, classes: tuple = None, input_dimentions=None):
        super(PyNet, self).__init__()
        if hidden_layers is None:
            return
        linear = []
        self.f = []

        # Add hidden layers
        for h in range(0, len(hidden_layers), 1):
            if h == 0:  # first hidden layer
                in_d = input_dimentions
                nonlinearity = Relu
            else:
                in_d = hidden_layers[h - 1]['neurons']
                nonlinearity = hidden_layers[h]['nonlinearity']
            out_d = hidden_layers[h]['neurons']
            linear.append(nn.Linear(in_d, out_d))
            if nonlinearity == Tanh:
                self.f.append(nn.Tanh())
            elif nonlinearity == Sigmoid:
                self.f.append(nn.Sigmoid())
            elif nonlinearity == Relu:
                self.f.append(nn.ReLU())
            elif nonlinearity == LeakyRelu:
                self.f.append(nn.LeakyReLU())
            else:
                pass

        # output layer
        linear.append(nn.Linear(hidden_layers[len(hidden_layers) - 1]['neurons'], len(classes)))
        # self.f.append(nn.ReLU())

        self.linear = nn.ModuleList(linear)

    def forward(self, X):
        x = X
        for i, l in enumerate(self.linear):
            if i == len(self.linear) - 1:
                x = l(x)
            else:
                x = self.f[i](l(x))
            # x = self.f[i](l(x))
        return x


class ANN:
    def __init__(self,
                 learning_rate=0.03,
                 hidden_layers=default_hidden_layers,
                 class_labels: tuple = None,
                 # classes=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
                 use_cpu=True,
                 process_as_a_batch=False,
                 optimizer_type=OP_TYPE_SGD,
                 loss_f=nn.CrossEntropyLoss(),
                 adwin_delta=1e-3,
                 back_prop_skip_loss_threshold=0.6):
        # configuration variables (which has the same name as init parameters)
        self.model_name = None
        self.learning_rate = learning_rate
        self.hidden_layers = copy.deepcopy(hidden_layers)
        self.class_labels = ('0', '1') if class_labels is None else class_labels
        self.use_cpu = use_cpu
        self.process_as_a_batch = process_as_a_batch
        self.optimizer_type = optimizer_type
        self.loss_f = loss_f
        self.adwin_delta = adwin_delta
        self.back_prop_skip_loss_threshold = back_prop_skip_loss_threshold

        # status variables
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.loss = None
        self.device = None
        self.samples_seen = 0
        self.trained_count = 0
        self.chosen_counts = 0
        self.estimator: BaseDriftDetector = None
        self.class_label_index_map = {}
        self.accumulated_loss = 0

        self.init_values()

    def init_values(self):
        # init status variables
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.loss = None
        self.device = None
        self.samples_seen = 0
        self.estimator = ADWIN(delta=self.adwin_delta)
        self.class_label_index_map = {}

        for i, class_label in enumerate(self.class_labels):
            self.class_label_index_map[class_label] = i

        if isinstance(self.hidden_layers, nn.Module):
            # assumes input dimention is set properly in the network structure
            self.net = copy.deepcopy(self.hidden_layers)
            self.initialize_net_para()
        elif isinstance(self.hidden_layers, list):
            if self.hidden_layers[0]['neurons'] is None or self.hidden_layers[0]['nonlinearity'] is None:
                print('Unknown hidden layer format is passed in: {}'.format(self.hidden_layers))
                print('Expected format :{}'.format(default_hidden_layers))
                exit(1)
        self.model_name = 'L1_L1n{}_{}_{:05f}_{}'.format(
            math.log(self.hidden_layers[0]['neurons'], 2) // 1 if self.hidden_layers[0]['neurons'] else 'NA',
            self.optimizer_type,
            self.learning_rate,
            self.adwin_delta)

        if self.use_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_optimizer(self):
        if self.optimizer_type == OP_TYPE_ADAGRAD or self.optimizer_type == OP_TYPE_ADAGRAD_NC:
            self.optimizer = optim.Adagrad(self.net.parameters(), lr=self.learning_rate, lr_decay=0, weight_decay=0,
                                           initial_accumulator_value=0, eps=1e-10)
        elif self.optimizer_type == OP_TYPE_ADADELTA or self.optimizer_type == OP_TYPE_ADADELTA_NC:
            self.optimizer = optim.Adadelta(self.net.parameters(), lr=self.learning_rate, eps=1e-10)
        elif self.optimizer_type == OP_TYPE_RMSPROP or self.optimizer_type == OP_TYPE_RMSPROP_NC:
            self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.learning_rate, alpha=0.99, weight_decay=0,
                                           eps=1e-10)
        elif self.optimizer_type == OP_TYPE_SGD or self.optimizer_type == OP_TYPE_SGD_NC:
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == OP_TYPE_ADAM or self.optimizer_type == OP_TYPE_ADAM_NC:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                        weight_decay=0, amsgrad=False)
        elif self.optimizer_type == OP_TYPE_ADAM_AMSG or self.optimizer_type == OP_TYPE_ADAM_AMSG_NC:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                        weight_decay=0, amsgrad=True)
        else:
            print('Invalid optimizer type = {}'.format(self.optimizer_type))

    def initialize_net_para(self):
        self.init_optimizer()
        print('Network configuration:\n'
              '{}\n'
              '======================================='.format(self))

    def initialize_network(self, input_dimentions=None):
        self.net = PyNet(hidden_layers=self.hidden_layers, classes=self.class_labels, input_dimentions=input_dimentions)
        self.initialize_net_para()

    def train_net(self, x, y):
        if torch.cuda.is_available():
            if self.device.type == 'cpu':
                pass
            else:
                x = x.to(self.device)
                y = y.to(self.device)
        else:
            pass

        self.optimizer.zero_grad()  # zero the gradient buffers
        # # forward propagation
        outputs = self.net(x)

        # backward propagation
        # print(self.net.linear[0].weight.data)
        class_index = self.class_label_index_map[y.reshape((-1,)).item()]
        # self.loss = self.criterion(outputs, y.reshape((-1,)).long())
        self.loss = self.loss_f(outputs, torch.tensor([class_index]))
        if self.loss.item() > self.back_prop_skip_loss_threshold:
            self.loss.backward()
            self.optimizer.step()  # Does the update
            self.trained_count += 1

        self.estimator.add_element(self.loss.item())
        self.accumulated_loss += self.loss.item()

        # if self.estimator.detected_change():
        #     print('drift detected by {}'.format(self.model_name))
        #     pass

    def partial_fit(self, X, r, c, y):
        if self.net is None:
            self.initialize_network(input_dimentions=c)

        if self.process_as_a_batch:
            self.samples_seen += r
            self.train_net(x=torch.from_numpy(X).float(), y=torch.from_numpy(np.array(y)).view(-1, 1).float())
        else:  # per instance processing (default behaviour)
            for i in range(r):
                x = torch.from_numpy(X[i])
                yy = torch.from_numpy(np.array(y[i]))
                x = x.view(1, -1).float()
                yy = yy.view(1, -1).float()
                x.unsqueeze(0)
                yy.unsqueeze(0)
                self.samples_seen += 1
                self.train_net(x=x, y=yy)

    def predict_proba(self, X, r, c):
        if self.net is None:
            self.initialize_network(input_dimentions=c)

        if self.process_as_a_batch:
            return self.net(torch.from_numpy(X).float()).detach().numpy()
        else:  # per instance processing (default behaviour)
            probas = None
            for i in range(r):
                x = torch.from_numpy(X[i])
                x = x.view(1, -1).float()
                x.unsqueeze(0)
                instance_class_probas = self.net(x).detach()
                if r == 1:
                    return instance_class_probas.reshape((1, -1))
                elif i == 0:
                    probas = instance_class_probas
                else:
                    probas = torch.cat(probas, instance_class_probas).detach()
            return probas

    def reset(self):
        # configuration variables (which has the same name as init parameters) should be copied by the caller function
        self.init_values()
        return self

    def get_loss_estimation(self):
        return self.estimator.estimation
        # return self.accumulated_loss/self.samples_seen if self.samples_seen != 0 else 0.0

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


def net_train(net: ANN, X: np.ndarray, r, c, y: np.ndarray):
    net.partial_fit(X, r, c, y)


class DeepNNPytorch(BaseSKMObject, ClassifierMixin):
    def __init__(self,
                 classes: tuple = None,
                 # classes=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
                 use_cpu=True,
                 process_as_a_batch=False,
                 use_threads=False,
                 stats_file=sys.stdout,
                 number_of_mlps_to_train=10,
                 number_of_instances_to_train_using_all_mlps_at_start=1000,
                 adwin_delta=1e-3,
                 back_prop_skip_loss_threshold=0.6,
                 stats_print_frequency=0):
        # configuration variables (which has the same name as init parameters)
        self.classes = ('0', '1') if classes is None else classes
        self.use_threads = use_threads
        self.stats_file = stats_file
        self.number_of_mlps_to_train = number_of_mlps_to_train
        self.number_of_instances_to_train_using_all_mlps_at_start = number_of_instances_to_train_using_all_mlps_at_start
        self.adwin_delta = adwin_delta
        self.stats_print_frequency = stats_print_frequency
        self.back_prop_skip_loss_threshold = back_prop_skip_loss_threshold

        super().__init__()

        # status variables
        self.nets = []  # type: List[ANN]
        self.heading_printed = False
        self.samples_seen = 0

        self.init_values()

    def init_values(self):
        # init status variables

        # init MLP pool
        for number_of_neurons_in_log2 in range(8, 11):
            for lr_denominator_in_log10 in range(1, 6):
                for optimizer_type in (OP_TYPE_SGD_NC, OP_TYPE_ADAM_NC):
                    self.nets.append(
                        ANN(hidden_layers=[{'neurons': 2 ** number_of_neurons_in_log2, 'nonlinearity': Relu}],
                            learning_rate=5 / (10 ** lr_denominator_in_log10),
                            optimizer_type=optimizer_type,
                            adwin_delta=self.adwin_delta,
                            back_prop_skip_loss_threshold=self.back_prop_skip_loss_threshold,
                            class_labels=self.classes))

        self.heading_printed = False
        self.stats_file = sys.stdout if self.stats_file is sys.stdout else open(self.stats_file, 'w')

        print(self)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        r, c = get_dimensions(X)
        t = []
        number_of_mlps_to_train = self.number_of_mlps_to_train
        number_of_top_mlps_to_train = self.number_of_mlps_to_train // 2

        if self.samples_seen < self.number_of_instances_to_train_using_all_mlps_at_start:
            number_of_mlps_to_train = len(self.nets)
            number_of_top_mlps_to_train = len(self.nets) // 2

        def custom_sort(k: ANN):
            return k.get_loss_estimation()

        self.nets.sort(key=custom_sort)

        for i in range(number_of_mlps_to_train):
            if i < number_of_top_mlps_to_train:
                # top most train
                nn_index = i
            else:
                # Random train
                off_set = ((self.samples_seen + i) % (len(self.nets) - number_of_top_mlps_to_train))
                nn_index = number_of_top_mlps_to_train + off_set
            if self.use_threads:
                t.append(threading.Thread(target=net_train, args=(self.nets[nn_index], X, r, c, y,)))
            else:
                net_train(self.nets[nn_index], X, r, c, y, )
        if self.use_threads:
            for i in range(len(t)):
                t[i].start()
            for i in range(len(t)):
                t[i].join()

        return self

    def get_chosen_index(self):
        chosen_index = 0
        min_estimation = sys.float_info.max
        for i in range(len(self.nets)):
            if round(self.nets[i].get_loss_estimation(), 2) < min_estimation:
                min_estimation = round(self.nets[i].get_loss_estimation(), 2)
                chosen_index = i
        return chosen_index

    def predict(self, X):
        r, c = get_dimensions(X)
        self.samples_seen += r
        current_best = self.get_chosen_index()
        self.nets[current_best].chosen_counts += 1
        _, predicted_idx = torch.max(self.nets[current_best].predict_proba(X, r, c), 1)
        self.print_stats()
        return [self.classes[predicted_idx.item()]]

    def predict_proba(self, X):
        pass

    def reset(self):
        # configuration variables (which has the same name as init parameters) should be copied by the caller function
        for i in range(len(self.nets)):
            self.nets[i].reset()
        return self

    def stream_ended(self):
        print('\nNetwork configuration:\n'
              '{}\n'
              '=======================================\n'
              'Foreground Nets\n'.format(self))
        self.print_stats(stream_ended=True)

    def print_stats(self, stream_ended=False):
        if self.samples_seen > 0 and (self.stats_print_frequency != 0 or stream_ended) and self.samples_seen % (
                self.stats_print_frequency if self.stats_print_frequency != 0 else self.samples_seen) == 0:
            if not self.heading_printed:
                print('id,'
                      'samples_seen,'
                      'trained_count,'
                      'optimizer_type_learning_rate_delta,'
                      'avg_loss,'
                      'estimated_loss,'
                      'chosen_counts',
                      file=self.stats_file)
                self.heading_printed = True
            for i in range(len(self.nets)):
                print('{},{},{},{},{},{},{}'.format(
                    self.samples_seen,
                    self.nets[i].samples_seen,
                    self.nets[i].trained_count,
                    self.nets[i].model_name,
                    self.nets[i].accumulated_loss/self.nets[i].samples_seen,
                    self.nets[i].estimator.estimation,
                    self.nets[i].chosen_counts), file=self.stats_file)
