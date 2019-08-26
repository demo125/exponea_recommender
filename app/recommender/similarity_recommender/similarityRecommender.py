import numpy as np
import pandas as pd
from recommender.similarity_recommender.combinedSimilarity import CombinedSimilarity
from recommender.recommenderABC import RecommenderABC
import os.path

class SimilarityRecommender(RecommenderABC):

    def __init__(self):
        self.combinedSimilarity = CombinedSimilarity()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(dir_path)
        self.events = pd.read_csv(dir_path + "'/../data/dataset_events.csv")
        self.products = pd.read_csv(dir_path + "/../data/dataset_catalog.csv")

    def train(self):
        self.combinedSimilarity.fetch_data()
        self.combinedSimilarity.preprocess()
        self.combinedSimilarity.train()

    def get_n_most_similar_products(self, product_id, n=50, exclude_products=[]):
        return self.combinedSimilarity.get_n_most_similar_products(product_id, n, exclude_products)

    def get_similarity_between_products(self, product_id1, product_id2):
        all_products = self.combinedSimilarity.get_n_most_similar_products(product_id1, n=99999999)
        idx = np.argwhere(np.array(all_products) == product_id2).flatten()[0] #O(n) can be optimized
        return all_similarities[idx]

    def can_recommend(self, customer_id):
        return self.events[self.events.customer_id == customer_id].shape[0] > 0
    
    def __get_similar_product_counts(self, n_customer_products, n_recommendations):
        n_cuts = np.repeat(n_recommendations // n_customer_products, n_customer_products)
        n_rest = n_recommendations % n_customer_products
        i = 0
        while n_rest != 0:
            n_cuts[i%len(n_cuts)] += 1
            n_rest -= 1
            i += 1

        return n_cuts

    def recommend(self, customer_id, n):
        customer_products = self.events[self.events.customer_id == customer_id].sort_values('timestamp', ascending=False).product_id.drop_duplicates().values.tolist()

        combined_similarity = np.zeros(self.products.shape[0])
        recommended_items = []
        n_cuts = self.__get_similar_product_counts(len(customer_products), n)

        n_most_products = []
        for i, product_id in enumerate(customer_products):

            n_most_products = self.get_n_most_similar_products(
                customer_id, 
                n_cuts[i], 
                recommended_items + customer_products #exclude already seen/bought/cart-added products, also exlude already receommended items
            )

            recommended_items += n_most_products
            
        return recommended_items