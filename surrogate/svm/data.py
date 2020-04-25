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

    def __init__(self, path, custom_data_dimensions=None):
        """
        class constructor
        :param path: file path to load the data
        :type path: str
        """
        super(SVCDataDenseBlock, self).__init__(path)
        self._constructed_data = None
        if custom_data_dimensions is None:
            self._data_dimensions = SVCDataDenseBlock.DATA_DIMENSIONS
        else:
            self._data_dimensions = custom_data_dimensions

    def save_svc_fitness_eval_data(self, fe_data):
        """
        Format fitness evaluation data and save the data to a csv file
        :param fe_data: data from fitness evaluation. The format is list[dict(block, losses, acc_history, acc_best)]
        :type fe_data: dict
        :return: whether it is save successfully
        :rtype: bool
        """
        modified_data = []
        for dict_row in fe_data:
            row_block = SVCDataBase._list_fill_none(dict_row['block'], self.block_dimensions)
            row_losses = SVCDataBase._list_fill_none(dict_row['losses'], self.losses_dimensions)
            row_acc_history = SVCDataBase._list_fill_none(dict_row['acc_history'], self.acc_history_dimensions)
            row = row_block + row_losses + row_acc_history + [dict_row['acc_best']]
            modified_data.append(row)
        header_block = SVCDataBase._generate_header_with_prefix('block', self.block_dimensions)
        header_losses = SVCDataBase._generate_header_with_prefix('losses', self.losses_dimensions)
        header_acc_history = SVCDataBase._generate_header_with_prefix('acc_history', self.acc_history_dimensions)
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
        header_losses = SVCDataBase._generate_header_with_prefix('losses', self.losses_dimensions)
        header_acc_history = SVCDataBase._generate_header_with_prefix('acc_history', self.acc_history_dimensions)
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
                    s_row = row_1.drop('acc_best').append(row_2.drop('acc_best')).append(pd.Series([flag]))
                    s_row = s_row.reset_index(drop=True)
                    df_constructed = df_constructed.append(s_row, ignore_index=True)
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
        if self.fe_data.loaded_data.empty:
            return pd.DataFrame()
        else:
            header_block = SVCDataBase._generate_header_with_prefix('block', self.block_dimensions)
            header_losses = SVCDataBase._generate_header_with_prefix('losses', epoch_extracted)
            header_acc_history = SVCDataBase._generate_header_with_prefix('acc_history', epoch_extracted)
            column_names = header_block + header_losses + header_acc_history + ['acc_best']
            return self.fe_data.loaded_data[column_names]

    @staticmethod
    def extract_fe_data_from_dataframe(df, epoch_extracted=10, block_dimensions=None):
        """
        extract block, losses and acc up to a certain number of epochs
        :param df: subset of the fitness evaluation data
        :type df: DataFrame
        :param epoch_extracted: number of epochs to extract losses and acc
        :type epoch_extracted: int
        :param block_dimensions: the dimensionality of the block config
        :type block_dimensions: int
        :return: extracted dataframe
        :rtype: DataFrame
        """
        block_dimensions = SVCDataDenseBlock.DATA_DIMENSIONS['block'] if block_dimensions is None else block_dimensions
        header_block = SVCDataBase._generate_header_with_prefix('block', block_dimensions)
        header_losses = SVCDataBase._generate_header_with_prefix('losses', epoch_extracted)
        header_acc_history = SVCDataBase._generate_header_with_prefix('acc_history', epoch_extracted)
        column_names = header_block + header_losses + header_acc_history
        return df[column_names]

    @property
    def constructed_data(self):
        """
        getter method of _constructed_data
        :return: the constructed data for svc model
        :rtype: DataFrame
        """
        return self._constructed_data

    @property
    def block_dimensions(self):
        return self._data_dimensions['block']

    @property
    def losses_dimensions(self):
        return self._data_dimensions['losses']

    @property
    def acc_history_dimensions(self):
        return self._data_dimensions['acc_history']

    @property
    def acc_best_dimensions(self):
        return self._data_dimensions['acc_best']
