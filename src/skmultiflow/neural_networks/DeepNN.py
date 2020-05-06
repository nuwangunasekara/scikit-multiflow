import numpy as np
from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils.utils import *

Sigmoid = 1
Tanh = 2
Relu = 3
LeakyRelu = 4

def derivative_of_relu(z):
    # g'(z) = 0 ; z < 0
    #       = 1 ; z >= 0
    return 1 if z >= 0 else 0


def derivative_of_leaky_relu(z):
    # g'(z) = 0.01 ; z < 0
    #       = 1 ; z >= 0
    return 1 if z >= 0 else 0.01


dev_of_relu = np.vectorize(derivative_of_relu)
dev_of_leaky_relu = np.vectorize(derivative_of_leaky_relu)

class DeepNN(BaseSKMObject, ClassifierMixin):
    """ Mask for sklearn.linear_model.Perceptron.

    scikit-multiflow requires a few interfaces, not present in scikit-learn,
    This mask serves as a wrapper for the Perceptron classifier.

    Examples
    --------
    .. code-block:: python

       # Imports
       from skmultiflow.neural_networks import PerceptronMask
       from skmultiflow.data import SEAGenerator

       # Setup a data stream
       stream = SEAGenerator(random_state=1)

       # Setup the Perceptron Mask
       perceptron = PerceptronMask()

       # Pre-train the Perceptron Mask with 200 samples

       # Prepare the processing of 5000 samples and correct prediction count
       n_samples = 0
       correct_cnt = 0
       while n_samples < 5000 and stream.has_more_samples():
           X, y = stream.next_sample()
           my_pred = perceptron.predict(X)
           if y[0] == my_pred[0]:
               correct_cnt += 1
           perceptron = perceptron.partial_fit(X, y, classes=stream.target_values)
           n_samples += 1

       # Display the results
       print('Perceptron Mask usage example')
       print('{} samples analyzed'.format(n_samples))
       print("Perceptron's performance: {}".format(correct_cnt / n_samples))
    """
    def __init__(self,
                 learning_rate = 0.8,
                 NNLayers=[{'neurons': 0, 'input_d': 3}, {'neurons': 4, 'g': Tanh}, {'neurons': 1, 'g': Sigmoid}]
                 ):
        self.learning_rate = learning_rate
        # TODO: deep copy maybe
        self.NNLayers = NNLayers

        super().__init__()

        for l in range(1, len(NNLayers), 1):
            if l == 1:
                self.NNLayers[l]['w'] = np.random.rand(self.NNLayers[l]['neurons'], self.NNLayers[0]['input_d']) * 0.01
            else:
                self.NNLayers[l]['w'] = np.random.rand(self.NNLayers[l]['neurons'], self.NNLayers[l - 1]['neurons']) * 0.01
            self.NNLayers[l]['b'] = np.zeros((self.NNLayers[l]['neurons'], 1))
            self.NNLayers[l]['z'] = np.zeros(self.NNLayers[l]['b'].shape)
            self.NNLayers[l]['a'] = np.zeros(self.NNLayers[l]['z'].shape)
            self.NNLayers[l]['dw'] = np.zeros(self.NNLayers[l]['w'].shape)
            self.NNLayers[l]['db'] = np.zeros(self.NNLayers[l]['b'].shape)
            self.NNLayers[l]['dz'] = np.zeros(self.NNLayers[l]['z'].shape)

        self.classes = [0,1]

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ partial_fit

        Calls the Perceptron partial_fit from sklearn.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The feature's matrix.

        y: Array-like
            The class labels for all samples in X.

        classes: Not used.

        sample_weight:
            Samples weight. If not provided, uniform weights are assumed.

        Returns
        -------
        PerceptronMask
            self

        """
        r, c = get_dimensions(X)

        for i in range(r):
            self.forward_prop(X[i])
            # print(y[i], self.predict(X))
            self.backward_prop(y[i])

        return self

    def predict(self, X):
        """ predict

        Uses the current model to predict samples in X.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The feature's matrix.

        Returns
        -------
        numpy.ndarray
            A numpy.ndarray containing the predicted labels for all instances in X.

        """

        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred

    def predict_proba(self, X):
        """ Predicts the probability of each sample belonging to each one of the known classes.
    
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.
    
        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, n_features), in which each outer entry is 
            associated with the X entry of the same index. And where the list in 
            index [i] contains len(self.target_values) elements, each of which represents
            the probability that the i-th sample of X belongs to a certain label.
    
        """
        proba = []
        r, c = get_dimensions(X)
        NNLayers = self.NNLayers

        for i in range(r):
            # self.forward_prop(X[i])
            proba.append([1 - NNLayers[len(NNLayers) - 1]['a'], NNLayers[len(NNLayers) - 1]['a']])

        return np.asarray(proba)

    def forward_prop(self, X):
        # forward propagation
        NNLayers = self.NNLayers
        NNLayers[0]['a'] = np.reshape(X, (NNLayers[0]['input_d'], 1))
        for l in range(1, len(NNLayers), 1):  # skip layer 0
            # z[l] = w[l].a[l-1]. + b[l]
            NNLayers[l]['z'] = np.dot(NNLayers[l]['w'], NNLayers[l - 1]['a']) + NNLayers[l]['b']
            # a[l] = g[l](z[l])
            if NNLayers[l]['g'] == Tanh:
                # a = e^z - e^-z / e^z + e^-z = e^2z - 1 / e^2z + 1
                NNLayers[l]['a'] = np.tanh(NNLayers[l]['z'])
            elif NNLayers[l]['g'] == Sigmoid:
                # a = 1 / 1 + e^-z
                NNLayers[l]['a'] = 1 / (1 + np.exp(-NNLayers[l]['z']))
            elif NNLayers[l]['g'] == Relu:
                # a = Max(0, z)
                NNLayers[l]['a'] = np.maximum(0, NNLayers[l]['z'])
            elif NNLayers[l]['g'] == LeakyRelu:
                # a = Max(0.01z, z)
                NNLayers[l]['a'] = np.maximum(0.01 * NNLayers[l]['z'], NNLayers[l]['z'])
            else:
                pass
            # if l == len(NNLayers) - 1:
            #     # Loss : L(y^, y) = -(y log(y^) + (1 - y)log(1 - y^))
            #     # cost[i]  = 1/i+1 SUM (L(y^, y)), i starts from 0
            #     J[i] = J[i - 1] - ((y * np.log(NNLayers[l]['a'])) + (
            #                 (1 - y) * np.log(1 - NNLayers[l]['a'])))  # each J[i]item will be divided by I[i] later

    def backward_prop(self, y):
        # backward propagation
        NNLayers = self.NNLayers
        # from last layer to 1 (if we go down till layer 0 (input) we could calculate dL/dx or dL/da[0])
        # for unsupervised learning
        for l in range(len(NNLayers) - 1, 0, -1):
            if l == (len(NNLayers) - 1):
                # da[L=last layer] = - y/a[L] + (1-y)/1-a[L]
                NNLayers[l]['da'] = (-y / NNLayers[l]['a']) + ((1 - y) / (1 - NNLayers[l]['a']))

            # dL/dz for other l th layers: dL/dz[l] = dz[l]
            # dz[l] = da[l] * g[l]'(z[l])
            if NNLayers[l]['g'] == Tanh:
                #                                      g'(z) = 1 - tanh(z)^2 = 1 - a
                NNLayers[l]['dz'] = NNLayers[l]['da'] * (1 - (NNLayers[l]['a'] ** 2))
            elif NNLayers[l]['g'] == Sigmoid:
                #                                      g'(z) = g(z) (1-g(z)) = a (1 - a)
                NNLayers[l]['dz'] = NNLayers[l]['da'] * (NNLayers[l]['a'] * (1 - NNLayers[l]['a']))
            elif NNLayers[l]['g'] == Relu:
                #                                      g'(z) = 0 ; z < 0, 1 ; z >= 0
                NNLayers[l]['dz'] = NNLayers[l]['da'] * dev_of_relu(NNLayers[l]['z'])
            elif NNLayers[l]['g'] == LeakyRelu:
                #                                      g'(z) = 0.01 ; z < 0, 1 ; z >= 0
                NNLayers[l]['dz'] = NNLayers[l]['da'] * dev_of_leaky_relu(NNLayers[l]['z'])
            else:
                pass

            # dw[l] =  dz[l].a[l-1]T
            NNLayers[l]['dw'] = np.dot(NNLayers[l]['dz'], NNLayers[l - 1]['a'].T)
            # db[l] =  dz[l]
            NNLayers[l]['db'] = NNLayers[l]['dz']

            NNLayers[l]['w'] -= (self.learning_rate * NNLayers[l]['dw'])
            NNLayers[l]['b'] -= (self.learning_rate * NNLayers[l]['db'])

            if l != 1:
                # dL/da or da for the l-1 layer
                # da[l-1] = w[l]T . dz[l]
                NNLayers[l - 1]['da'] = np.dot(NNLayers[l]['w'].T, NNLayers[l]['dz'])