"""
file for the functions related to manipulate data for svm surrogate model
"""

from edllib.evo import FitnessEvaluationData
import pandas as pd
from pandas import DataFrame


class SVCDataBase:
    def __init__(self, path):
        self._fe_data = FitnessEvaluationData(path)

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
    def _generate_header_with_prefix(prefix, length):
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

    @property
    def fe_data(self):
        """
        getter method for fitness evaluation data
        :return: _fe_data
        :rtype: FitnessEvaluationData
        """
        return self._fe_data


class SVCDataDenseBlock(SVCDataBase):
    DATA_DIMENSIONS = {'block': 32,
                       'losses': 100,
                       'acc_history': 100,
                       'acc_best': 1}
    """
    SVC data class for evolving dense block
    """

    def __init__(self, path):
        """
        class constructor
        :param path: file path to load the data
        :type path: str
        """
        super(SVCDataDenseBlock, self).__init__(path)
        self._constructed_data = None

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
        header_block = SVCDataBase._generate_header_with_prefix('block', SVCDataDenseBlock.DATA_DIMENSIONS['block'])
        header_losses = SVCDataBase._generate_header_with_prefix('losses', SVCDataDenseBlock.DATA_DIMENSIONS['losses'])
        header_acc_history = SVCDataBase._generate_header_with_prefix('acc_history',
                                                                      SVCDataDenseBlock.DATA_DIMENSIONS['acc_history'])
        column_names = header_block + header_losses + header_acc_history + ['acc_best']
        return self.fe_data.save_or_append_data(modified_data, column_names)

    def search_svc_fitness_eval_data_by_block(self, block_config):
        """
        search fitness evaluation data by the given block config
        :param block_config: the block config
        :type block_config: list
        :return: the search result
        :rtype: DataFrame
        """
        header_losses = SVCDataBase._generate_header_with_prefix('losses', SVCDataDenseBlock.DATA_DIMENSIONS['losses'])
        header_acc_history = SVCDataBase._generate_header_with_prefix('acc_history',
                                                                      SVCDataDenseBlock.DATA_DIMENSIONS['acc_history'])
        column_names = header_losses + header_acc_history + ['acc_best']
        return self.fe_data.search_by_features(block_config, excluded_columns=column_names)

    def construct_svc_data(self, epoch_extracted=10):
        """
        construct dataframe ready to be fed to SVC
        :param epoch_extracted: number of epochs to extract the training data
        :type epoch_extracted: int
        :return: constructed data
        :rtype: DataFrame
        """
        df_constructed = pd.DataFrame([])
        self.fe_data.load_data(True)
        df_extracted = self._extract_fe_data(epoch_extracted)
        for index_1, row_1 in df_extracted.iterrows():
            for index_2, row_2 in df_extracted.iterrows():
                if not row_1.equals(row_2):
                    flag = 1 if row_1['acc_best'] < row_2['acc_best'] else 0
                    s_row = row_1.drop('acc_best').append(row_2.drop('acc_best')).reset_index(drop=True)
                    s_row.append([flag])
                    df_constructed.append(s_row, ignore_index=True)
        self._constructed_data = df_constructed
        return self._constructed_data

    def _extract_fe_data(self, epoch_extracted=10):
        """
        extract block, losses and acc up to a certain number of epochs, and best acc
        :param epoch_extracted: number of epochs to extract losses and acc
        :type epoch_extracted: int
        :return: extracted dataframe
        :rtype: DataFrame
        """
        header_block = SVCDataBase._generate_header_with_prefix('block', SVCDataDenseBlock.DATA_DIMENSIONS['block'])
        header_losses = SVCDataBase._generate_header_with_prefix('losses', epoch_extracted)
        header_acc_history = SVCDataBase._generate_header_with_prefix('acc_history', epoch_extracted)
        column_names = header_block + header_losses + header_acc_history + ['acc_best']
        return self.fe_data.loaded_data[column_names]

    @property
    def constructed_data(self):
        """
        getter method of _constructed_data
        :return: the constructed data for svc model
        :rtype: DataFrame
        """
        return self._constructed_data
