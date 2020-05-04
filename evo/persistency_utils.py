import pickle


class PopulationPersistent:
    def __init__(self):
        None

    @staticmethod
    def save_populiation(pop, path, protocol=pickle.DEFAULT_PROTOCOL):
        """
        save a population using pickle
        :param pop: the population to be saved
        :type pop: object
        :param path: path to write the serialised population
        :type path: string
        :param protocol: pkl protocol
        :type protocol: int
        """
        with open(path, 'wb') as output:
            pickle.dump(pop, output, protocol)
        output.close()

    @staticmethod
    def load_pop(path):
        """
        load population from persisted pickle file
        :param path: pickle file path
        :type path: string
        :return: loaded population
        :rtype: object
        """
        with open(path, 'rb') as input:
            pop = pickle.load(input)
        input.close()
        return pop

    @staticmethod
    def convert_pop_pkl_protocol(path, protocol=pickle.DEFAULT_PROTOCOL):
        """
        load pop and save to a specified pkl protocol version
        to solve compatibility issue
        :param path: pkl file path
        :type path: str
        :param protocol: pkl protocol
        :type protocol: int
        """
        pop = PopulationPersistent.load_pop(path)
        PopulationPersistent.save_populiation(pop, path, protocol)


class IndividualPersistent:
    def __init__(self):
        None

    @staticmethod
    def save_individual(ind, path):
        """
        save a individual using pickle
        :param ind: the individual to be saved
        :type ind: object
        :param path: path to write the serialised individual
        :type path: string
        """
        with open(path, 'wb') as output:
            pickle.dump(ind, output, pickle.DEFAULT_PROTOCOL)
        output.close()

    @staticmethod
    def load_individual(path):
        """
        load individual from persisted pickle file
        :param path: pickle file path
        :type path: string
        :return: loaded individual
        :rtype: object
        """
        with open(path, 'rb') as input:
            ind = pickle.load(input)
        input.close()
        return ind
