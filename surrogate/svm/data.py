"""
file for the functions related to manipulate data for svm surrogate model
"""

from edllib.evo import FitnessEvaluationData


class SVCDataBase:
    def __init__(self):
        None

    @staticmethod
    def _list_fill_none(lt, length, filling=None):
        """
        Extend list to a certain length
        :param lt: list to be extended
        :type lt: list
        :param length: the length that the list needs to be extended to
        :type length: int
        :return: the filled list
        :rtype: list
        """
        filled_list = [lt[i] if i < len(lt) else filling for i in range(length)]
        return filled_list

    @staticmethod
    def _generate_header_with_suffix(prefix, length):
        """
        Generate headers by using the same prefix
        :param prefix: prefix of the header names
        :type prefix: str
        :param length: the number of header names to generate
        :type length: int
        :return: the generated headers
        :rtype: list
        """
        lt_header = [prefix + '_' + str(i) for i in range(length)]
        return lt_header


class SVCDataDenseBlock(SVCDataBase):
    DATA_DIMENSIONS = {'block': 32,
                       'losses': 100,
                       'acc_history': 100,
                       'acc_best': 1}
    """
    SVC data class
    """

    def __init__(self, path):
        """
        class constructor
        :param path: file path to load the data
        :type path: str
        """
        super(SVCDataDenseBlock, self).__init__()
        self.fe_data = FitnessEvaluationData(path)
        self.constructed_data = None

    def save_svc_fitness_eval_data(self, fe_data):
        """
        Format fitness evaluation data and save the data to a csv file
        :param fe_data: data from fitness evaluation. The format is list[dict(block, losses, acc_history, acc_best)]
        :type fe_data: list
        :return: whether it is save successfully
        :rtype: bool
        """
        modified_data = []
        for dict_row in fe_data:
            row_block = SVCDataBase._list_fill_none(dict_row['block'], SVCDataDenseBlock.DATA_DIMENSIONS['block'])
            row_losses = SVCDataBase._list_fill_none(dict_row['losses'], SVCDataDenseBlock.DATA_DIMENSIONS['losses'])
            row_acc_history = SVCDataBase._list_fill_none(dict_row['acc_history'],
                                                          SVCDataDenseBlock.DATA_DIMENSIONS['acc_history'])
            row = row_block + row_losses + row_acc_history + dict_row['acc_best']
            modified_data.append(row)
        header_block = SVCDataBase._generate_header_with_suffix('block', SVCDataDenseBlock.DATA_DIMENSIONS['block'])
        header_losses = SVCDataBase._generate_header_with_suffix('losses', SVCDataDenseBlock.DATA_DIMENSIONS['losses'])
        header_acc_history = SVCDataBase._generate_header_with_suffix('acc_history',
                                                                      SVCDataDenseBlock.DATA_DIMENSIONS['acc_history'])
        column_names = header_block + header_losses + header_acc_history + ['acc_best']
        return self.fe_data.save_or_append_data(modified_data, column_names)

    def construct_svc_data(self):
        None
