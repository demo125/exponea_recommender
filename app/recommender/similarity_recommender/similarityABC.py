from abc import ABC, abstractmethod
import pandas as pd
import os.path
import pickle

class SimilarityABC(ABC):

    def __init__(self, weight, name='ABC'):
        self.weight = weight
        self.name = name
        self.data = None
        self.preprocessed_data = None
        self.model = None

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = dir_path + '/../data/'

    def fetch_data(self):
        self.data = self.do_fetch_data()
        print(self.name + ": data fetched")
        
    def get_weight(self):
        return self.weight

    @abstractmethod
    def do_fetch_data(self):
        pass

    def preprocess(self):
        if self.data is None:
            self.fetch_data()

        self.preprocessed_data = self.do_preprocess()
        print(self.name + ": data preprocessed")
        self._export()

    @abstractmethod
    def do_preprocess(self):
        pass

    def _export(self):
        with open(self.data_dir + self.name + ".pickle", 'wb') as f:
            pickle.dump(self.preprocessed_data, f)
            f.close()
            print(self.name + ": data saved")

    def _import_data(self):
        with open(self.data_dir + self.name + '.pickle', 'rb') as f:
            self.preprocessed_data = pickle.load(f)
            f.close()
            print(self.name+ ': data loaded')

    def train(self):
        if not os.path.isfile(self.data_dir + self.name + '.pickle'):
            self.preprocess()

        self._import_data()

        self.model = self.do_train()
        print(self.name + ": trained")
    
    @abstractmethod
    def do_train(self):
        pass

    def get_product_similarities(self, product_id):
        if self.model is None: self.train()
        return self.do_get_product_similarities(product_id)
    
    @abstractmethod
    def do_get_product_similarities(self, product_id):
        pass

    def is_product_id_valid(self, product_id):
        if self.preprocessed_data is None: self.train()
        product_idx = product_id - 1
        return product_idx >= 0 and product_idx < self.preprocessed_data.shape[0]
    