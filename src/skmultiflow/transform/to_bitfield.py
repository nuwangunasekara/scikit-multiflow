import numpy as np

from skmultiflow.transform.base_transform import StreamTransform
from skmultiflow.utils.utils import get_dimensions


class ToBitField(StreamTransform):

    def __init__(self):
        super().__init__()

    def transform(self, X):
        r, c = get_dimensions(X)

        custom_dtype_b = np.dtype([('bytes', np.uint8, 8)])
        X_u4 = X.view(custom_dtype_b)
        X_u4_u = np.unpackbits(X_u4['bytes'], axis=1)
        return X_u4_u.reshape(r, c * 8 * 8)

    def fit(self, X, y):
        return self

    def partial_fit_transform(self, X, y=None, classes=None):
        return self.transform(X)

    def partial_fit(self, X, y=None, classes=None):
        return self
