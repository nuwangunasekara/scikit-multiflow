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

default_network_layers = [{'neurons': 0, 'input_d': 0}, {'neurons': 2 ** 10, 'g': 3}, {'neurons': 1, 'g': 1}]

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


class ANN:
    def __init__(self,
                 learning_rate=0.03,
                 network_layers=default_network_layers,
                 class_labels=['0','1'],  # {'up':0,'down':1}
                 use_cpu=True,
                 process_as_a_batch=False,
                 optimizer_type=OP_TYPE_SGD,
                 warning_detection_method: BaseDriftDetector = ADWIN(delta=1e-8, direction=ADWIN.DETECT_DOWN),
                 drift_detection_method: BaseDriftDetector = ADWIN(delta=1e-3, direction=ADWIN.DETECT_DOWN)):
        # configuration variables (which has the same name as init parameters)
        self.learning_rate = learning_rate
        self.network_layers = copy.deepcopy(network_layers)
        self.class_labels = class_labels
        self.use_cpu = use_cpu
        self.process_as_a_batch = process_as_a_batch
        self.optimizer_type = optimizer_type
        if self.optimizer_type == OP_TYPE_SGD_NC \
                or self.optimizer_type == OP_TYPE_ADAGRAD_NC \
                or self.optimizer_type == OP_TYPE_RMSPROP_NC \
                or self.optimizer_type == OP_TYPE_ADADELTA_NC \
                or self.optimizer_type == OP_TYPE_ADAM_NC \
                or self.optimizer_type == OP_TYPE_ADAM_AMSG_NC:
            self.drift_detection_method = None
            self.warning_detection_method = None
        else:
            self.drift_detection_method = drift_detection_method
            if self.drift_detection_method.__class__.__name__ == 'HDDM_A' \
                    or self.drift_detection_method.__class__.__name__ == 'HDDM_W':
                if warning_detection_method is not None:
                    print('Parameter warning_detection_method should be None for drift_detection_methods HDDM_A and'
                          ' HDDM_W as they have in built warning detection. Hence setting it to None.')
                    self.warning_detection_method = None
                else:
                    self.warning_detection_method = None
            else:
                self.warning_detection_method = warning_detection_method

        # status variables
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.loss = None
        self.device = None
        self.class_to_label = {}
        self.label_to_class = {}
        self.samples_seen = 0
        self.detected_warnings = 0
        self.accumulated_loss = 0
        self.accumulated_loss_since_last_detected_drift_by_parent = 0
        self.samples_seen_after_last_detected_drift_by_parent = 0
        self.last_detected_drift_by_parent_at = 0

        self.init_values()

    def init_values(self):
        # init status variables
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.loss = None
        self.device = None
        self.class_to_label = {}
        self.label_to_class = {}
        self.samples_seen = 0
        self.detected_warnings = 0
        self.accumulated_loss = 0
        self.accumulated_loss_since_last_detected_drift_by_parent = 0
        self.samples_seen_after_last_detected_drift_by_parent = 0
        self.last_detected_drift_by_parent_at = 0

        initialize_network = False

        for i in range(len(self.class_labels)):
            self.class_to_label.update({i: self.class_labels[i]})
            self.label_to_class.update({self.class_labels[i]: i})

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
            self.network_layers = copy.deepcopy(default_network_layers)
            print('Unknown network type passed in, set the network type to default: {}'.format(self.network_layers))

        if initialize_network:
            self.initialize_network(self.network_layers)

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
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-10,
                                        weight_decay=0, amsgrad=False)
        elif self.optimizer_type == OP_TYPE_ADAM_AMSG or self.optimizer_type == OP_TYPE_ADAM_AMSG_NC:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-10,
                                        weight_decay=0, amsgrad=True)
        else:
            print('Invalid optimizer type = {}'.format(self.optimizer_type))

    def initialize_net_para(self):
        self.init_optimizer()

        # for multi class classification
        # criterion = nn.CrossEntropyLoss()
        # for binary classification
        # combines a Sigmoid layer
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.BCELoss()
        print('Network configuration:\n'
              '{}\n'
              '======================================='.format(self))

    def initialize_network(self, network_layers=None):
        self.net = PyNet(network_layers)
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
        output = self.net(x)

        # backward propagation
        # print(self.learning_rate)
        # print(self.net.linear[0].weight.data)
        if self.learning_rate > 0.0:
            # print('here')
            self.loss = self.criterion(output, y)
            self.loss.backward()
            self.optimizer.step()  # Does the update
            self.accumulated_loss += self.loss.item()
            self.accumulated_loss_since_last_detected_drift_by_parent += self.loss.item()

        labels_proba = torch.cat((1 - output, output), 1).detach()
        # TODO: we may have to have a special case for batch processing
        predicted_labels = torch.argmax(labels_proba, dim=1).detach()

        if self.drift_detection_method is not None:
            # get predicted class and compare with actual class label
            predicted_matches_actual = predicted_labels == y
            self.drift_detection_method.add_element(1 if predicted_matches_actual else 0)
            if self.warning_detection_method is not None:
                self.warning_detection_method.add_element(1 if predicted_matches_actual else 0)

            # pass the difference to the detector
            # predicted_matches_actual = torch.abs(y-output).detach().numpy()[0]
            # self.drift_detection_method.add_element(predicted_matches_actual)

            # Check if the was a warning
            if self.warning_detection_method is not None:
                if self.warning_detection_method.detected_change():
                    self.detected_warnings += 1
            else:  # warning detector is None, hence drift detector has warning detection capability.
                if self.drift_detection_method.detected_warning_zone():
                    self.detected_warnings += 1  # 3 is the threshold level
            # Check if the was a change
            if self.detected_warnings > 3 and self.drift_detection_method.detected_change():
                print('Drift detected by {} around {} th sample. Hence resetting optimizer'.format(
                    self.drift_detection_method, self.samples_seen))
                self.detected_warnings = 0
                self.init_optimizer()

        return labels_proba, predicted_labels

    def partial_fit(self, X, r, c, y, last_detected_drift_by_parent_at):
        if self.last_detected_drift_by_parent_at < last_detected_drift_by_parent_at:
            self.accumulated_loss_since_last_detected_drift_by_parent = 0
            self.last_detected_drift_by_parent_at = last_detected_drift_by_parent_at
        if self.net is None:
            self.network_layers[0]['input_d'] = c
            self.initialize_network(self.network_layers)

        if self.process_as_a_batch:
            self.samples_seen += r
            self.samples_seen_after_last_detected_drift_by_parent += r
            # probas, y_hats are still tensors
            probas, y_hats = self.train_net(x=torch.from_numpy(X).float(), y=torch.from_numpy(np.array(y)).view(-1, 1).float())
            return probas.numpy(), y_hats.numpy(), self.accumulated_loss_since_last_detected_drift_by_parent/self.samples_seen_after_last_detected_drift_by_parent
        else:  # per instance processing (default behaviour)
            probas = None
            y_hats = None
            for i in range(r):
                x = torch.from_numpy(X[i])
                yy = torch.from_numpy(np.array(y[i]))
                x = x.view(1, -1).float()
                yy = yy.view(1, -1).float()
                x.unsqueeze(0)
                yy.unsqueeze(0)
                self.samples_seen += 1
                self.samples_seen_after_last_detected_drift_by_parent += 1
                proba, y_hat = self.train_net(x=x, y=yy)
                if i == 0:
                    probas = proba.detach().clone()
                    y_hats = y_hat.detach().clone()
                else:
                    torch.cat(probas, proba, dim=0, out=probas)
                    torch.cat(y_hats, y_hat, dim=0, out=y_hats)
            return probas.numpy(), y_hats.numpy(), self.accumulated_loss_since_last_detected_drift_by_parent/self.samples_seen_after_last_detected_drift_by_parent

    def predict_proba(self, X, r, c):
        # r, c = get_dimensions(X)

        if self.net is None:
            self.network_layers[0]['input_d'] = c
            self.initialize_network(self.network_layers)

        if self.process_as_a_batch:
            y_prob = self.net(torch.from_numpy(X).float())
            return torch.cat((1 - y_prob, y_prob), 1).detach().numpy()
        else:  # per instance processing (default behaviour)
            proba = []
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

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


def net_train(net: ANN, X: np.ndarray, r, c, y: np.ndarray, train_results, i, last_detected_drift_by_parent_at):
    train_results['probas'][i], train_results['y_hats'][i], train_results['avg_loss_since_last_detected_drift_by_parent'][i] = net.partial_fit(X, r, c, y, last_detected_drift_by_parent_at)


def net_predict_proba(net: ANN, X: np.ndarray, r, c, probas, i):
    probas[i] = net.predict_proba(X, r, c)


foreground_net_config = [
    {'optimizer_type': OP_TYPE_SGD_NC, 'l_rate': 0.03},
    {'optimizer_type': OP_TYPE_SGD_NC, 'l_rate': 0.07},
    {'optimizer_type': OP_TYPE_RMSPROP_NC, 'l_rate': 0.01},
    {'optimizer_type': OP_TYPE_ADAGRAD_NC, 'l_rate': 0.03},
    {'optimizer_type': OP_TYPE_ADAGRAD_NC, 'l_rate': 0.09},
    {'optimizer_type': OP_TYPE_ADAM_NC, 'l_rate': 0.01},
]

background_net_config = [
    {'optimizer_type': OP_TYPE_SGD_NC, 'l_rate': 0.05},
    {'optimizer_type': OP_TYPE_ADAGRAD_NC, 'l_rate': 0.07},
    {'optimizer_type': OP_TYPE_ADAGRAD, 'l_rate': 0.09},
]


class DeepNNPytorch(BaseSKMObject, ClassifierMixin):
    def __init__(self,
                 class_labels=['0','1'],  # {'up':0,'down':1}
                 use_cpu=True,
                 process_as_a_batch=False,
                 use_threads=False,
                 background_training_after=4):
        # configuration variables (which has the same name as init parameters)
        self.class_labels = class_labels
        self.use_threads = use_threads
        self.background_training_after = background_training_after

        super().__init__()

        # status variables
        self.class_to_label = {}
        self.foreground_nets = []  # type: List[ANN]
        self.background_nets = []  # type: List[ANN]
        self.drift_detection_method = None
        self.warning_detection_method = None
        self.detected_warnings = 0
        self.samples_seen = 0
        self.last_detected_drift_around = 0
        self.background_learner_threads = []
        self.background_train_results = None
        self.foreground_train_results = None

        self.init_status_values()

    def init_status_values(self):
        # init status variables
        self.class_to_label = {}
        for i in range(len(self.class_labels)):
            self.class_to_label.update({i: self.class_labels[i]})

        for i in range(len(foreground_net_config)):
            self.foreground_nets.append(ANN(learning_rate=foreground_net_config[i]['l_rate'], optimizer_type=foreground_net_config[i]['optimizer_type'],
                                            class_labels=self.class_labels))

        for i in range(len(background_net_config)):
            self.background_nets.append(ANN(learning_rate=foreground_net_config[i]['l_rate'], optimizer_type=background_net_config[i]['optimizer_type'],
                                            class_labels=self.class_labels))

        self.drift_detection_method = ADWIN(delta=1e-3, direction=ADWIN.DETECT_DOWN)
        self.warning_detection_method = ADWIN(delta=1e-8, direction=ADWIN.DETECT_DOWN)

        self.detected_warnings = 0
        self.samples_seen = 0
        self.last_detected_drift_around = 0
        self.background_learner_threads = []
        self.background_train_results = None
        self.foreground_train_results = None
        print(self)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        r, c = get_dimensions(X)
        self.samples_seen += r

        # if self.samples_seen % 2 == 0:
        if len(self.background_learner_threads) == 0:
            if self.samples_seen % self.background_training_after == 0:
                self.background_train_results = {'probas': [None] * len(self.background_nets),
                                                 'y_hats': [None] * len(self.background_nets),
                                                 'avg_loss_since_last_detected_drift_by_parent': [0] * len(
                                                     self.background_nets)}
                for i in range(len(self.background_nets)):
                    self.background_learner_threads.append(threading.Thread(target=net_train,
                                                                            args=(self.background_nets[i], X, r, c, y, self.background_train_results, i, self.last_detected_drift_around,)))

                for i in range(len(self.background_nets)):
                    self.background_learner_threads[i].start()
        else:  # there are live background learner threads
            # wait for self.background_training_after instances to join them
            if self.samples_seen % self.background_training_after == self.background_training_after - 1:
                # TODO: CPython does not support multi threading: https://docs.python.org/3/library/threading.html
                #  we still may be fine as long as we don't compile the module using CPython.
                #  Multiprocessing is an alternative:
                #  https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing
                for i in range(len(self.background_nets)):
                    self.background_learner_threads[i].join()
                self.background_learner_threads = []

                if self.foreground_train_results is not None:
                    min_back = np.argmin(self.background_train_results['avg_loss_since_last_detected_drift_by_parent'], axis=0)
                    max_fore = np.argmax(self.foreground_train_results['avg_loss_since_last_detected_drift_by_parent'], axis=0)
                    # min_back < max_fore
                    if self.background_train_results['avg_loss_since_last_detected_drift_by_parent'][min_back] \
                            < self.foreground_train_results['avg_loss_since_last_detected_drift_by_parent'][max_fore]:
                        tmp_net = self.foreground_nets[max_fore]
                        self.foreground_nets[max_fore] = self.background_nets[min_back]
                        self.background_nets[min_back] = tmp_net

        self.foreground_train_results = {'probas': [None] * len(self.foreground_nets),
                                         'y_hats': [None] * len(self.foreground_nets),
                                         'avg_loss_since_last_detected_drift_by_parent': [0] * len(
                                             self.foreground_nets)}
        if self.use_threads:
            t = []
            for i in range(len(self.foreground_nets)):
                t.append(threading.Thread(target=net_train, args=(self.foreground_nets[i], X, r, c, y, self.foreground_train_results, i, self.last_detected_drift_around,)))

            for i in range(len(self.foreground_nets)):
                t[i].start()

            for i in range(len(self.foreground_nets)):
                t[i].join()
        else:
            for i in range(len(self.foreground_nets)):
                net_train(self.foreground_nets[i], X, r, c, y, self.foreground_train_results, i, self.last_detected_drift_around)

        if self.drift_detection_method is not None:
            # get predicted class and compare with actual class label
            predicted_label = vectorized_map_class_to_label(np.argmax(np.sum(self.foreground_train_results['probas'], axis=0) / len(self.foreground_nets), axis=1),
                                                            class_to_label_map=self.class_to_label)
            # TODO: we may have to have a special case for batch processing
            predicted_matches_actual = predicted_label == y

            self.drift_detection_method.add_element(1 if predicted_matches_actual else 0)
            if self.warning_detection_method is not None:
                self.warning_detection_method.add_element(1 if predicted_matches_actual else 0)

            # pass the difference to the detector
            # predicted_matches_actual = torch.abs(y-output).detach().numpy()[0]
            # self.drift_detection_method.add_element(predicted_matches_actual)

            # Check if the was a warning
            if self.warning_detection_method is not None:
                if self.warning_detection_method.detected_change():
                    self.detected_warnings += 1
            else:  # warning detector is None, hence drift detector has warning detection capability.
                if self.drift_detection_method.detected_warning_zone():
                    self.detected_warnings += 1  # 3 is the threshold level
            # Check if the was a change
            if self.detected_warnings > 3 and self.drift_detection_method.detected_change():
                print('Drift detected by {} around {} th sample.'.format(
                    self.drift_detection_method, self.samples_seen))
                self.detected_warnings = 0
                self.last_detected_drift_around = self.samples_seen
                # Find the the worst learner from the foreground and replace it with the background

        return self

    def predict(self, X):
        y_proba = self.predict_proba(X)
        pred_sum_per_class = np.sum(y_proba, axis=0)
        pred_avgsum_per_class = np.divide(pred_sum_per_class, len(self.foreground_nets))
        y_pred = np.argmax(pred_avgsum_per_class, axis=0)
        return vectorized_map_class_to_label(np.asarray([y_pred]), class_to_label_map=self.class_to_label)

    def predict_proba(self, X):
        r, c = get_dimensions(X)
        probas = np.zeros([len(self.foreground_nets), len(self.class_labels)])
        # if self.use_threads:
        #     t = []
        #     for i in range(len(self.nets)):
        #         t.append(threading.Thread(target=net_predict_proba, args=(self.nets[i], X, r, c, probas, i,)))
        #
        #     for i in range(len(self.nets)):
        #         t[i].start()
        #
        #     for i in range(len(self.nets)):
        #         t[i].join()
        # else:
        for i in range(len(self.foreground_nets)):
            net_predict_proba(self.foreground_nets[i], X, r, c, probas, i)

        return np.asarray(probas)

    def reset(self):
        # configuration variables (which has the same name as init parameters) should be copied by the caller function
        for i in range(len(self.foreground_nets)):
            self.foreground_nets[i].reset()
        return self

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def stream_ended(self):
        print('\nNetwork configuration:\n'
              '{}\n'
              '=======================================\n'
              'Foreground Nets\n'.format(self))
        print('optimizer_type,learning_rate,accumulated_loss,accumulated_loss_since_last_detected_drift_by_parent')
        for i in range(len(self.foreground_nets)):
            print('{},{},{},{}'.format(
                self.foreground_nets[i].optimizer_type,
                self.foreground_nets[i].learning_rate,
                self.foreground_nets[i].accumulated_loss/self.foreground_nets[i].samples_seen,
                self.foreground_nets[i].accumulated_loss_since_last_detected_drift_by_parent/self.foreground_nets[i].samples_seen_after_last_detected_drift_by_parent))
        print('\n'
              'Background Nets\n'.format(self))
        for i in range(len(self.background_nets)):
            print('{},{},{},{}'.format(
                self.background_nets[i].optimizer_type,
                self.background_nets[i].learning_rate,
                self.background_nets[i].accumulated_loss / self.background_nets[i].samples_seen,
                self.background_nets[i].accumulated_loss_since_last_detected_drift_by_parent / self.background_nets[
                    i].samples_seen_after_last_detected_drift_by_parent))
        print('\n')
