from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils.utils import *
from skmultiflow.neural_networks.utils import *


class DeepNN(BaseSKMObject, ClassifierMixin):
    def __init__(self,
                 learning_rate=0.1,
                 network_layers=None,
                 class_labels=['0','1'],  # {'up':0,'down':1}
                 input_dimensions=None):

        initialize_network = False
        self.network_initialized = False
        self.network_config = []
        self.learning_rate = learning_rate
        self.class_to_label = {}
        self.label_to_class = {}
        for i in range(len(class_labels)):
            self.class_to_label.update({i: class_labels[i]})
            self.label_to_class.update({class_labels[i]: i})
        print('class_to_label=', self.class_to_label)
        print('label_to_class=', self.label_to_class)

        self.layer = None
        if network_layers is None:
            if input_dimensions is None:
                # wait till we receive the first instance to get input dimensions to initialize a basic network
                pass
            else:
                network_layers = auto_configure_layers(input_dimensions=input_dimensions, return_basic=True)
                self.network_config = network_layers
                initialize_network = True
        else:
            if network_layers[0]['input_d'] is None or network_layers[0]['input_d'] == 0:
                if input_dimensions is None:
                    # wait till we receive the first instance to get input dimensions
                    # to initialize the passed-in network
                    pass
                else:
                    network_layers[0]['input_d'] = input_dimensions
                    initialize_network = True
            else:
                initialize_network = True
            self.network_config = network_layers

        if initialize_network:
            self.initialize_network(network_layers=network_layers)

        super().__init__()

    def initialize_network(self, network_layers=None):
        if network_layers is None:
            return
        # TODO: deep copy maybe
        self.layer = network_layers
        for l in range(1, len(network_layers), 1):
            if l == 1:
                self.layer[l]['w'] = np.random.rand(self.layer[l]['neurons'], self.layer[0]['input_d']) * 0.01
            else:
                self.layer[l]['w'] = np.random.rand(self.layer[l]['neurons'],
                                                    self.layer[l - 1]['neurons']) * 0.01
            self.layer[l]['b'] = np.zeros((self.layer[l]['neurons'], 1))
            self.layer[l]['z'] = np.zeros(self.layer[l]['b'].shape)
            self.layer[l]['a'] = np.zeros(self.layer[l]['z'].shape)
            self.layer[l]['dw'] = np.zeros(self.layer[l]['w'].shape)
            self.layer[l]['db'] = np.zeros(self.layer[l]['b'].shape)
            self.layer[l]['dz'] = np.zeros(self.layer[l]['z'].shape)
        self.network_initialized = True

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        r, c = get_dimensions(X)

        if not self.network_initialized:
            if len(self.network_config) > 0:
                self.network_config[0]['input_d'] = c
            else:
                self.network_config = auto_configure_layers(input_dimensions=c, return_basic=True)
            self.initialize_network(network_layers=self.network_config)

        for i in range(r):
            self.forward_prop(X[i])
            self.backward_prop(self.label_to_class[y[i]])
        return self

    def predict(self, X):
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
        return vectorized_map_class_to_label(y_pred, class_to_label_map=self.class_to_label)

    def predict_proba(self, X):
        proba = []
        r, c = get_dimensions(X)

        if not self.network_initialized:
            if len(self.network_config) > 0:
                self.network_config[0]['input_d'] = c
            else:
                self.network_config = auto_configure_layers(input_dimensions=c, return_basic=True)
            self.initialize_network(network_layers=self.network_config)

        for i in range(r):
            self.forward_prop(X[i])
            proba.append([1 - self.layer[len(self.layer) - 1]['a'], self.layer[len(self.layer) - 1]['a']])
        return np.asarray(proba)

    def forward_prop(self, X):
        # forward propagation
        self.layer[0]['a'] = np.reshape(X, (self.layer[0]['input_d'], 1))
        for l in range(1, len(self.layer), 1):  # skip layer 0
            # z[l] = w[l].a[l-1]. + b[l]
            self.layer[l]['z'] = np.dot(self.layer[l]['w'], self.layer[l - 1]['a']) + self.layer[l]['b']
            # a[l] = g[l](z[l])
            if self.layer[l]['g'] == Tanh:
                # a = e^z - e^-z / e^z + e^-z = e^2z - 1 / e^2z + 1
                self.layer[l]['a'] = np.tanh(self.layer[l]['z'])
            elif self.layer[l]['g'] == Sigmoid:
                # a = 1 / 1 + e^-z
                self.layer[l]['a'] = 1 / (1 + np.exp(-self.layer[l]['z']))
            elif self.layer[l]['g'] == Relu:
                # a = Max(0, z)
                self.layer[l]['a'] = np.maximum(0, self.layer[l]['z'])
            elif self.layer[l]['g'] == LeakyRelu:
                # a = Max(0.01z, z)
                self.layer[l]['a'] = np.maximum(0.01 * self.layer[l]['z'], self.layer[l]['z'])
            else:
                pass
            # if l == len(self.layer) - 1:
            #     # Loss : L(y^, y) = -(y log(y^) + (1 - y)log(1 - y^))
            #     # cost[i]  = 1/i+1 SUM (L(y^, y)), i starts from 0
            #     J[i] = J[i - 1] - ((y * np.log(self.layer[l]['a'])) + (
            #                 (1 - y) * np.log(1 - self.layer[l]['a'])))  # each J[i]item will be divided by I[i] later

    def backward_prop(self, y):
        # backward propagation
        # from last layer to 1 (if we go down till layer 0 (input) we could calculate dL/dx or dL/da[0])
        # for unsupervised learning
        for l in range(len(self.layer) - 1, 0, -1):
            if l == (len(self.layer) - 1):
                # da[L=last layer] = - y/a[L] + (1-y)/1-a[L]
                self.layer[l]['da'] = (-y / self.layer[l]['a']) + ((1 - y) / (1 - self.layer[l]['a']))

            # dL/dz for other l th layers: dL/dz[l] = dz[l]
            # dz[l] = da[l] * g[l]'(z[l])
            if self.layer[l]['g'] == Tanh:
                #                                      g'(z) = 1 - tanh(z)^2 = 1 - a
                self.layer[l]['dz'] = self.layer[l]['da'] * (1 - (self.layer[l]['a'] ** 2))
            elif self.layer[l]['g'] == Sigmoid:
                #                                      g'(z) = g(z) (1-g(z)) = a (1 - a)
                self.layer[l]['dz'] = self.layer[l]['da'] * (self.layer[l]['a'] * (1 - self.layer[l]['a']))
            elif self.layer[l]['g'] == Relu:
                #                                      g'(z) = 0 ; z < 0, 1 ; z >= 0
                self.layer[l]['dz'] = self.layer[l]['da'] * dev_of_relu(self.layer[l]['z'])
            elif self.layer[l]['g'] == LeakyRelu:
                #                                      g'(z) = 0.01 ; z < 0, 1 ; z >= 0
                self.layer[l]['dz'] = self.layer[l]['da'] * dev_of_leaky_relu(self.layer[l]['z'])
            else:
                pass

            # dw[l] =  dz[l].a[l-1]T
            self.layer[l]['dw'] = np.dot(self.layer[l]['dz'], self.layer[l - 1]['a'].T)
            # db[l] =  dz[l]
            self.layer[l]['db'] = self.layer[l]['dz']

            self.layer[l]['w'] -= (self.learning_rate * self.layer[l]['dw'])
            self.layer[l]['b'] -= (self.learning_rate * self.layer[l]['db'])

            if l != 1:
                # dL/da or da for the l-1 layer
                # da[l-1] = w[l]T . dz[l]
                self.layer[l - 1]['da'] = np.dot(self.layer[l]['w'].T, self.layer[l]['dz'])