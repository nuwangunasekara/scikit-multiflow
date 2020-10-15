import numpy as np

from skmultiflow.transform.base_transform import StreamTransform
# from skmultiflow.trees.nodes.active_learning_node_perceptron import compute_sd
from skmultiflow.utils.utils import get_dimensions


def compute_sd(sum_of_squares: float, sum_of_values: float, samples_seen: float):
    variance = 0.0
    if samples_seen > 1:
        variance = (sum_of_squares - ((sum_of_values * sum_of_values) / samples_seen)) / samples_seen
    return np.sqrt(variance)


# def add_values(sum_of_attribute_values, sample_weight, x):
#     sum_of_values = 0.0
#     try:
#         sum_of_values = np.add(sum_of_attribute_values, np.multiply(sample_weight, x))
#     except ValueError:
#         sum_of_values = np.multiply(sample_weight, x)
#     return sum_of_values
#
#
# def add_squares(sum_of_attribute_squares, sample_weight, x):
#     sum_of_squares = 0.0
#     try:
#         sum_of_squares = np.add(sum_of_attribute_squares, np.multiply(sample_weight, np.power(x, 2)))
#     except ValueError:
#         sum_of_squares = np.multiply(sample_weight, np.power(x, 2))
#     return sum_of_squares


def normalize_sample(mean, sd, x):
    normalized_sample = 0.0
    if sd > 0.0:
        normalized_sample = float(x - mean) / (3 * sd)
        # normalized_sample = float(x - mean) / (sd)
    return normalized_sample


v_compute_sd = np.vectorize(compute_sd)
# v_add_values = np.vectorize(add_values)
# v_add_squares = np.vectorize(add_squares)
v_normalize_sample = np.vectorize(normalize_sample)


class Normalize(StreamTransform):

    def __init__(self):
        # print(__name__, 1)
        self.seen_at_least_one_sample = False
        self.samples_seen = []
        self.feature_weights = []
        self.sum_of_attribute_values = []
        self.sum_of_attribute_squares = []
        super().__init__()

    def initialize(self, X):
        r, c = get_dimensions(X)
        self.feature_weights = np.ones((1, c))
        self.samples_seen = np.zeros((1, c))
        self.sum_of_attribute_values = np.zeros((1, c))
        self.sum_of_attribute_squares = np.zeros((1, c))

    def transform(self, X):
        # print(__name__, 2)
        self._partial_fit(X)
        normalized_samples = []
        if self.seen_at_least_one_sample:
            r, c = get_dimensions(X)
            for i in range(r):
                # XX = np.array(X[i])
                # if samples_seen > 1: # This is always True as we come here ONLY if seen_at_least_one_sample is True
                mean = self.sum_of_attribute_values[i] / self.samples_seen[i]
                sd = v_compute_sd(self.sum_of_attribute_squares[i], self.sum_of_attribute_values[i], self.samples_seen[i])
                b = v_normalize_sample(mean, sd, X[i])
                b = np.reshape(b, -1)
                normalized_samples.append(b)
        else:
            pass
        normalized_samples = np.reshape(normalized_samples, -1)
        normalized_samples = [normalized_samples]
        return np.array(normalized_samples)

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

    def _partial_fit(self, X):
        # print(__name__, 6)
        row_cnt, _ = get_dimensions(X)
        if not self.seen_at_least_one_sample and row_cnt > 0:
            self.initialize(X)
            self.seen_at_least_one_sample = True
        for i in range(row_cnt):
            self.samples_seen[i] += 1
            # self.sum_of_attribute_values[i] = v_add_values(self.sum_of_attribute_values[i], self.feature_weights, X[i])
            # self.sum_of_attribute_squares[i] = v_add_squares(self.sum_of_attribute_squares[i], self.feature_weights, X[i])
            try:
                self.sum_of_attribute_values[i] = np.add(self.sum_of_attribute_values[i],
                                                         np.multiply(self.feature_weights, X[i]))
                self.sum_of_attribute_squares[i] = np.add(self.sum_of_attribute_squares[i],
                                                          np.multiply(self.feature_weights, np.power(X[i], 2)))
            except ValueError:
                self.sum_of_attribute_values[i] = np.multiply(self.feature_weights, X[i])
                self.sum_of_attribute_squares[i] = np.multiply(self.feature_weights, np.power(X[i], 2))
