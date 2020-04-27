"""
File to develop surrogate models
"""
import numpy as np
from .data import SVCDataDenseBlock
from sklearn import svm, preprocessing
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split


class SVCModelBase:
    """
    SVC surrogate model base class
    """

    def __init__(self, svc_data, kernel='rbf', threshold=0.8, norm=True):
        """
        class constructor
        :param svc_data: data to fit svc model. Dict with 2-d list X and 1-d list y
        :type svc_data: dict
        :param kernel: kernel function of the svc model
        :type kernel: str
        :param threshold: threshold to enable prediction
        :type threshold: float
        :param norm: normalize data or not
        :type norm: bool
        """
        self._svc_data = svc_data
        self._kernel = kernel
        self._svc_model = svm.SVC(kernel='rbf', gamma='scale')
        self._scores = None
        self._threshold = threshold
        self._norm = norm
        self._scaler = self._init_scaler() if self._norm else None

    def _init_scaler(self):
        """
        init scaler for normalizing data
        """
        self._scaler = preprocessing.MinMaxScaler().fit(self._svc_data['X'])
        return self._scaler

    def calc_selection_score(self, cv=None, test_size=0.2):
        """
        calculate the score for the model based on the given dataset
        :param cv: the number of fold
        :type cv: int
        :param test_size: test data ration when train/test validation is used
        :type test_size: float
        :return: the scores as an array
        :rtype: np.array
        """
        data_X = self.scaler.transform(self.svc_data['X']) if self._norm else self.svc_data['X']
        data_y = self.svc_data['y']
        if cv is None:
            self._scores = self.train_test_val_score(data_X, data_y, test_size)
        else:
            self._scores = cross_val_score(self.svc_model, data_X, data_y, cv=cv)
        return self._scores

    def train_test_val_score(self, data_X, data_y, test_size=0.2):
        """
        Use train/test split to validate the model
        :param data_X: feature data. 2-d array
        :type data_X: array
        :param data_y: label data. 1-d array
        :type data_y: array
        :param test_size: test data ratio
        :type test_size: float
        :return: scores of the evaluation
        :rtype: array
        """
        d_train_X, d_test_X, d_train_y, d_test_y = train_test_split(data_X, data_y, test_size=test_size,
                                                                    random_state=1)
        self.svc_model.fit(d_train_X, d_train_y)
        d_predict_y = self.svc_model.predict(d_test_X)
        acc = np.equal(d_test_y, d_predict_y).sum() / d_test_y.shape[0]
        return np.array([acc])

    def fit(self, calc_scores=True, cv=None):
        """
        fit svc model
        :param calc_scores: whether calculating scores when fitting the svc model
        :type calc_scores: bool
        :param cv: the number of fold
        :type cv: int
        """
        self.calc_selection_score(cv=cv) if calc_scores is True else None
        data_X = self.scaler.transform(self.svc_data['X']) \
            if self._norm is None else self.svc_data['X']
        self.svc_model.fit(data_X, self.svc_data['y'])

    def predict(self, data_X):
        """
        predict by using svc model
        :param data_X: feature data, which can be 1-d or 2-d array
        :type data_X: array
        :return: predicted labels
        :rtype: array
        """
        y = None
        if self.is_activated():
            data_X = np.array(data_X)
            data_X = np.reshape(data_X, (1, data_X.shape[0])) if len(data_X.shape) == 1 else data_X
            data_X = self.scaler.transform(data_X) if self._norm else data_X
            y = self.svc_model.predict(data_X)
        return y

    def is_activated(self):
        """
        check whether the surrogate model is activated (>threshold)
        :return: activated or not
        :rtype: bool
        """
        if self._scores is not None and self.scores.mean() > self._threshold:
            return True
        else:
            return False

    @property
    def svc_model(self):
        """
        getter method for _svc_model
        :return: svc model
        :rtype: SVC
        """
        return self._svc_model

    @property
    def scores(self):
        """
        getter method for _scores
        :return: scores of the svc model
        :rtype: array
        """
        return self._scores

    @property
    def svc_data(self):
        """
        getter method for _svc_data
        :return: data to fit svc model. Dict with 2-d list X and 1-d list y
        :rtype: dict
        """
        return self._svc_data

    @svc_data.setter
    def svc_data(self, svc_data):
        """
        setter method for _svc_data
        :param svc_data: data to fit svc model. Dict with 2-d list X and 1-d list y
        :type svc_data: dict
        """
        self._svc_data = svc_data

    @property
    def scaler(self):
        """
        getter method for _scaler
        :return: scaler for normalizing data
        :rtype: MinMaxScaler
        """
        return self._scaler


class SVCModelDenseBlock(SVCModelBase):
    """
    SVC surrogate model for evolving dense block
    """

    def __init__(self, data, kernel='rbf', threshold=0.8, norm=True):
        """
        class constructor
        :param data: data to fit svc model
        :type data: SVCDataDenseBlock
        :param kernel: kernel function of the svc model
        :type kernel: str
        :param threshold: threshold to enable prediction
        :type threshold: float
        :param norm: normalize data or not
        :type norm: bool
        """
        self._data = data
        svc_data = self.convert_data_to_svc_data()
        super(SVCModelDenseBlock, self).__init__(svc_data, kernel, threshold, norm)

    def convert_data_to_svc_data(self):
        """
        convert fitness evaluation data to data fed to the svc model
        :return: data to be fed to the svc model, which is a dict contains X and y
        :rtype: dict
        """
        constructed_svc_data = self.data.construct_svc_data()
        if constructed_svc_data.empty:
            svc_data = {}
        else:
            data_X = constructed_svc_data.iloc[:, :-1].to_numpy()
            data_y = constructed_svc_data.iloc[:, -1].to_numpy()
            svc_data = {'X': data_X, 'y': data_y}
        return svc_data

    def reload_svc_data(self):
        """
        reload data and construct svc data
        """
        svc_data = self.convert_data_to_svc_data()
        self.svc_data = svc_data

    @property
    def data(self):
        """
        getter method for _data
        :return: _data
        :rtype: SVCDataDenseBlock
        """
        return self._data
