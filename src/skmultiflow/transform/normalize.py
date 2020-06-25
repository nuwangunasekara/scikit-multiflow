import numpy as np

from skmultiflow.transform.base_transform import StreamTransform
# from skmultiflow.trees.nodes.active_learning_node_perceptron import compute_sd
from skmultiflow.utils.utils import get_dimensions


def compute_sd(square_val: float, val: float, size: float):
    if size > 1:
        a = square_val - ((val * val) / size)
        if a > 0:
            return np.sqrt(a / size)
    return 0.0


class Normalize(StreamTransform):

    def __init__(self,
                 nominal_attributes=None):
        # print(__name__, 1)
        self._nominal_attributes = nominal_attributes
        self.samples_seen = 0
        self.sum_of_attribute_values = []
        self.sum_of_attribute_squares = []
        super().__init__()

    def transform(self, X):
        # print(__name__, 2)
        self._partial_fit(X)
        normalized_samples = []
        if self.samples_seen > 0:
            r, c = get_dimensions(X)
            for i in range(r):
                normalized_samples.append(self.normalize_sample(X[i]))
        else:
            pass
        return np.array(normalized_samples)

    def normalize_sample(self, X):
        normalized_sample = []
        for i in range(len(X)):
            if (self._nominal_attributes is None or (self._nominal_attributes is not None and
                                                     i not in self._nominal_attributes)) and \
                    self.samples_seen > 1:
                mean = self.sum_of_attribute_values[i] / self.samples_seen
                sd = compute_sd(self.sum_of_attribute_squares[i], self.sum_of_attribute_values[i],
                                self.samples_seen)
                if sd > 0:
                    normalized_sample.append(float(X[i] - mean) / (3 * sd))
                else:
                    normalized_sample.append(0.0)
            else:
                normalized_sample.append(0.0)
        # if self.samples_seen > 1:
        #     normalized_sample.append(1.0)  # Value to be multiplied with the constant factor
        # else:
        #     normalized_sample.append(0.0)
        return np.asarray(normalized_sample)

    def fit(self, X, y):
        # print(__name__, 3)
        return self

    def partial_fit_transform(self, X, y=None, classes=None):
        # print(__name__, 4)
        return self.transform(X)

    def partial_fit(self, X, y=None, classes=None):
        # print(__name__, 5)
        self._partial_fit(X)
        return self

    def _partial_fit(self, X
                     # , y
                     # , sample_weight=None
                     ):
        # print(__name__, 6)
        # if y is not None:
        sample_weight = None
        row_cnt, _ = get_dimensions(X)
        if sample_weight is None:
            # sample_weight = 1
            sample_weight = np.ones(row_cnt)
            # sample_weight = np.ones((1, row_cnt))
        if row_cnt != len(sample_weight):
            raise ValueError('Inconsistent number of instances ({}) and weights ({}).'.
                             format(row_cnt, len(sample_weight)))
        for i in range(row_cnt):
            if sample_weight[i] != 0.0:
                self.samples_seen += sample_weight[i]
                try:
                    self.sum_of_attribute_values = np.add(self.sum_of_attribute_values,
                                                          np.multiply(sample_weight[i], X[i]))
                    self.sum_of_attribute_squares = np.add(
                        self.sum_of_attribute_squares, np.multiply(sample_weight[i], np.power(X[i], 2))
                    )
                except ValueError:
                    self.sum_of_attribute_values = np.multiply(sample_weight[i], X[i])
                    self.sum_of_attribute_squares = np.multiply(sample_weight[i], np.power(X[i], 2))
