import pandas as pd
import numpy as np
from recommender.similarity_recommender.categorySimilarity import CategorySimilarity
from recommender.similarity_recommender.brandSimilarity import BrandSimilarity
from recommender.similarity_recommender.genderSimilarity import GenderSimilarity
from recommender.similarity_recommender.priceSimilarity import PriceSimilarity

class CombinedSimilarity():

    def __init__(self):

        classes = {
            "CategorySimilarity":CategorySimilarity,
            "BrandSimilarity": BrandSimilarity,
            "GenderSimilarity": GenderSimilarity,
            "PriceSimilarity": PriceSimilarity
        }

        weights = {
            "CategorySimilarity": 0.2,
            "BrandSimilarity": 0.1,
            "GenderSimilarity": 0.5,
            "PriceSimilarity": 0.2
        }

        self.similarities = dict()
        for name, classs in classes.items():
            self.similarities[name] = classs(weights[name])
        self.trained = False

    def fetch_data(self):
        for name, sim in self.similarities.items():
            sim.fetch_data()
        print('similarity calculator: data fetched')

    def preprocess(self):
        for name, sim in self.similarities.items():
            sim.preprocess()
        print('similarity calculator: preprocessed')

    def train(self):
        for name, sim in self.similarities.items():
            sim.train()
            self.trained = True
        print('similarity calculator: trained')

    def _is_product_id_valid(self, product_id):
        valid = True
        for name, sim in self.similarities.items():
            valid = sim.is_product_id_valid(product_id) and valid
        return valid

    def get_n_most_similar_products(self, product_id, n=20, exclude_products=[]):
        if not self._is_product_id_valid(product_id): return []
        similarities = dict()
        product_count = 0
        for name, sim in self.similarities.items():
            similarities[name] = sim.get_product_similarities(product_id) 
            product_count = len(similarities[name])
        
        combined_similarity = np.zeros(product_count)
        for name in similarities:
            combined_similarity += similarities[name] * self.similarities[name].get_weight()
        
        combined_similarity /= np.max(combined_similarity)

        if len(exclude_products):
            exclude_products_idxs = np.array(exclude_products) - 1
            combined_similarity[exclude_products_idxs] = - 1
        
        sorted_idxs = np.argsort(combined_similarity)[::-1][:n]
        n_most_similar_products = sorted_idxs + 1
        similarities = combined_similarity[sorted_idxs]

        return n_most_similar_products[:n].tolist()#,  similarities.tolist()