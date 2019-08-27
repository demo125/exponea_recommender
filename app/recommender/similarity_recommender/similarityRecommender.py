import numpy as np
import pandas as pd
from recommender.similarity_recommender.combinedSimilarity import CombinedSimilarity
from recommender.recommenderABC import RecommenderABC
import os.path

#preprocessing and training logic is described in notebooks/2_similarity_recommender.ipynb
class SimilarityRecommender(RecommenderABC):

    #initialize model that combines different feature similarities
    def __init__(self):
        self.combinedSimilarity = CombinedSimilarity()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.events = pd.read_csv(dir_path + "'/../data/dataset_events.csv")
        self.products = pd.read_csv(dir_path + "/../data/dataset_catalog.csv")

    #runs all methods needed for recommender to make recommendations
    def train(self):
        self.combinedSimilarity.fetch_data()
        self.combinedSimilarity.preprocess()
        self.combinedSimilarity.train()

    def get_n_most_similar_products(self, product_id, n=50, exclude_products=[]):
        return self.combinedSimilarity.get_n_most_similar_products(product_id, n, exclude_products)


    def get_similarity_between_products(self, product_id1, product_id2):
        all_products, all_similarities = self.combinedSimilarity.get_n_most_similar_products(product_id1, n=99999999, return_similarities=True)
        idx = np.argwhere(np.array(all_products) == product_id2).flatten()[0] #O(n) can be optimized
        return all_similarities[idx]

    #recommendations are possible if customer visited at least single product - customer_id is in dataset_events
    def can_recommend(self, customer_id):
        return self.events[self.events.customer_id == customer_id].shape[0] > 0
    
    #returns list of counts of how many similar items of already "interacted" products should be choosen as recommendation
    def __get_similar_product_counts(self, n_customer_products, n_recommendations):
        n_counts = np.repeat(n_recommendations // n_customer_products, n_customer_products)
        n_rest = n_recommendations % n_customer_products
        i = 0
        while n_rest != 0:
            n_counts[i%len(n_counts)] += 1
            n_rest -= 1
            i += 1
        return n_counts

    #making recommendations
    def recommend(self, customer_id, n):
        #selecting already "interacted" products and sorting them by "interaction" time
        customer_products = self.events[self.events.customer_id == customer_id].sort_values('timestamp', ascending=False).product_id.drop_duplicates().values.tolist()

        combined_similarity = np.zeros(self.products.shape[0])
        recommended_items = []
        n_counts = self.__get_similar_product_counts(len(customer_products), n)

        n_most_products = []
        #iterate over "interacted" products
        for i, product_id in enumerate(customer_products):

            #finding n_counts[i] similar products to "iteracted" product
            n_most_products = self.get_n_most_similar_products(
                product_id, 
                n_counts[i], 
                recommended_items + customer_products #exclude already "interacted" products, also exlude already receommended items
            ) 

            recommended_items += n_most_products 
            
        return recommended_items