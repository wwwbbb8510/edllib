"""
data utils where data related tools for EDL are developed
"""

import pandas as pd
from pandas import DataFrame
import os


class FileStorage:
    """
    Base class of file storage
    """

    def __init__(self, path):
        """
        class constructor
        :param path: full path to store the data
        :type path: str
        """
        self._path = path
        self._loaded_data = None
        self.load_data()

    def save_or_append_data(self, data, column_names, file_type='csv'):
        """
        save data if the file doesn't exist; otherwise, append the data
        :param data: data to save or append, which is 2-d list
        :type data: list
        :param column_names: column names
        :type column_names: list
        :param file_type: file type to store the data
        :type file_type: str
        :return: whether it's save successfully
        :rtype: bool
        """
        if not os.path.exists(self.path):
            save_result = self.save_data(data, column_names, file_type)
        else:
            save_result = self.append_data(data, column_names, file_type)
        return save_result

    def save_data(self, data, column_names, file_type='csv'):
        """
        save data of a 2-d list to file
        :param data: data to be saved, which is a 2-d list
        :type data: list
        :param column_names: column names of the data, which is a 1-d list
        :type column_names: list
        :param file_type: the file type for the file to store the data, default:csv
        :type file_type: str
        :return: whether the data is saved successfully
        :rtype: bool
        """
        is_success = False
        df_data = pd.DataFrame(data, columns=column_names)
        if file_type == 'csv':
            try:
                df_data.to_csv(self.path, index=False)
                is_success = True
            except:
                is_success = False
        return is_success

    def append_data(self, data, column_names, file_type='csv'):
        """
        append data of a 2-d list to file
        :param data: data to be saved, which is a 2-d list
        :type data: list
        :param column_names: column names of the data, which is a 1-d list
        :type column_names: list
        :param file_type: the file type for the file to store the data, default:csv
        :type file_type: str
        :return: whether the data is saved successfully
        :rtype: bool
        """
        is_success = False
        df_data = pd.DataFrame(data, columns=column_names)
        if file_type == 'csv':
            try:
                df_data.to_csv(self.path, mode='a', index=False, header=False)
                is_success = True
            except:
                is_success = False
        return is_success

    def search_by_features(self, row, excluded_columns=[]):
        """
        search a row from the loaded data
        :param row: the data used to be searched, which is a 1-d list
        :type row: list
        :param excluded_columns: column names of non features, e.g. class labels
        :type excluded_columns: list
        :return: the search result
        :rtype: DataFrame
        """
        df_result = pd.DataFrame([])
        if not self.loaded_data.empty:
            list_query = []
            val_index = 0
            for col_name in self.loaded_data.columns:
                if col_name not in excluded_columns:
                    col_query = col_name + '==' + str(row[val_index])
                    list_query.append(col_query)
                    val_index += 1
            if len(list_query) > 0:
                if len(list_query) > 16:
                    df_result = self.loaded_data.query(' and '.join(list_query[0:16]))
                    df_result = df_result.query(' and '.join(list_query[16:]))
                else:
                    df_result = self.loaded_data.query(' and '.join(list_query))

        return df_result

    def load_data(self, reload=False):
        """
        load data from csv
        :param reload: force to reload
        :type reload: bool
        """
        if self.loaded_data is None or reload is True:
            try:
                self._loaded_data = pd.read_csv(self.path)
            except:
                self._loaded_data = pd.DataFrame()

    def _filter_data_append(self, data, excluded_columns=[]):
        """
        Filter data to append to avoid duplicated records
        :param data:
        :type data:
        :return:
        :rtype:
        """
        filtered_data = []
        self.load_data(reload=True)
        if not self.loaded_data.empty:
            for row in data:
                df_result = self.search_by_features(row, excluded_columns)
                if df_result.empty:
                    filtered_data.append(row)
        else:
            filtered_data = data
        return filtered_data

    @property
    def path(self):
        """
        getter method for _path
        :return: path
        :rtype: str
        """
        return self._path
    
    @property
    def loaded_data(self):
        """
        getter method for _loaded_data
        :return: the loaded data
        :rtype: DataFrame
        """
        return self._loaded_data


class FitnessEvaluationData(FileStorage):
    """
    Save data for Surrogate SVC
    """

    def __init__(self, path):
        """
        class constructor
        :param path: full path to store data
        :type path: str
        """
        super(FitnessEvaluationData, self).__init__(path)
