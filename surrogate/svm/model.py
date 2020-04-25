"""
File to develop surrogate models
"""

from .data import SVCDataDenseBlock
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


class SVCModelBase:
    """
    SVC surrogate model base class
    """

    def __init__(self, svc_data, kernel='rbf', threshold=0.8):
        """
        class constructor
        :param svc_data: data to fit svc model. Dict with 2-d list X and 1-d list y
        :type svc_data: dict
        :param kernel: kernel function of the svc model
        :type kernel: str
        :param threshold: threshold to enable prediction
        :type threshold: float
        """
        self._svc_data = svc_data
        self._kernel = kernel
        self._svc_model = svm.SVC(kernel='rbf', gamma='scale')
        self._scores = None
        self._threshold = threshold

    def calc_selection_score(self, cv=10):
        """
        calculate the score for the model based on the given dataset
        :param cv: the number of fold
        :type cv: int
        :return: the scores as an array
        :rtype: np.array
        """
        self._scores = cross_val_score(self.svc_model, self.svc_data['X'], self.svc_data['y'], cv=cv)
        return self._scores

    def fit(self, calc_scores=True, cv=10):
        """
        fit svc model
        :param calc_scores: whether calculating scores when fitting the svc model
        :type calc_scores: bool
        :param cv: the number of fold
        :type cv: int
        """
        self.calc_selection_score(cv=cv) if calc_scores is True else None
        self.svc_model.fit(self.svc_data['X'], self.svc_data['y'])

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


class SVCModelDenseBlock(SVCModelBase):
    """
    SVC surrogate model for evolving dense block
    """

    def __init__(self, data, kernel='rbf', threshold=0.8):
        """
        class constructor
        :param data: data to fit svc model
        :type data: SVCDataDenseBlock
        :param kernel: kernel function of the svc model
        :type kernel: str
        :param threshold: threshold to enable prediction
        :type threshold: float
        """
        self._data = data
        svc_data = self.convert_data_to_svc_data()
        super(SVCModelDenseBlock, self).__init__(svc_data, kernel, threshold)

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
