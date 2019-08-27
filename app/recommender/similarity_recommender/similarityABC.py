from abc import ABC, abstractmethod
import pandas as pd
import os.path
import pickle

#preprocessing and training logic is described in notebooks/2_similarity_recommender.ipynb
class SimilarityABC(ABC):

    def __init__(self, weight, name='ABC'):
        self.weight = weight
        self.name = name
        self.data = None
        self.preprocessed_data = None
        self.model = None

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = dir_path + '/../data/'

    #returns weight of the feature
    def get_weight(self):
        return self.weight
    
    #runs abstract method do_fetch_data on inherited class which returns raw data
    #in the future every class can fetch data from diferent sources
    def fetch_data(self):
        self.data = self.do_fetch_data()
        print(self.name + ": data fetched")

    @abstractmethod
    def do_fetch_data(self):
        pass

    #check if data were fetch, if not fetches the data
    #runs preprocessing method do_preprocess and store preprocessed data 
    def preprocess(self):
        if self.data is None:
            self.fetch_data()

        self.preprocessed_data = self.do_preprocess()
        print(self.name + ": data preprocessed")
        self._export() #exports data

    @abstractmethod
    def do_preprocess(self):
        pass

    #preprocessing can take time, so we can store preprocessed data in binary file
    def _export(self):
        with open(self.data_dir + self.name + ".pickle", 'wb') as f:
            pickle.dump(self.preprocessed_data, f)
            f.close()
            print(self.name + ": data saved")

    #improting preprocessed data from binary file
    def _import_data(self):
        with open(self.data_dir + self.name + '.pickle', 'rb') as f:
            self.preprocessed_data = pickle.load(f)
            f.close()
            print(self.name+ ': data loaded')

    #check if data were preprocessed and preprocess them if needed
    #import preprocessed data and trains the model
    def train(self):
        if not os.path.isfile(self.data_dir + self.name + '.pickle'):
            self.preprocess()

        self._import_data()

        self.model = self.do_train()
        print(self.name + ": trained")
    
    @abstractmethod
    def do_train(self):
        pass

    #trains the model if its not trained, returns list of similarities ordered by product ids
    def get_product_similarities(self, product_id):
        if self.model is None: self.train()
        return self.do_get_product_similarities(product_id)
    
    @abstractmethod
    def do_get_product_similarities(self, product_id):
        pass

    #check if product id is valid
    def is_product_id_valid(self, product_id):
        if self.preprocessed_data is None: self.train()
        product_idx = product_id - 1
        return product_idx >= 0 and product_idx < self.preprocessed_data.shape[0]
    