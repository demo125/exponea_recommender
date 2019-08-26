from recommender.similarity_recommender.similarityRecommender import SimilarityRecommender
from recommender.popular_items_recommender.popRecommender import PopRecommender
from recommender.light_fm_recommender.lightFmRecommender import LightFmRecommender
import numpy as np

class Controller:

    def __init__(self):
        self.similarityRecommender = SimilarityRecommender()
        self.popRecommender = PopRecommender()
        self.lightFmRecommender = LightFmRecommender()

    def train(self):
        self.similarityRecommender.train()
        self.popRecommender.train()
        self.lightFmRecommender.train()
        print("all trained")

    def get_n_most_similar_products(self, product_id, n):
        return self.similarityRecommender.get_n_most_similar_products(product_id, n)

    def get_similarity_between_products(self, product_id1, product_id2):
        return self.similarityRecommender.get_similarity_between_products(product_id1, product_id2)

    def get_recommendations(self, customer_id, n):
        if self.lightFmRecommender.can_recommend(customer_id):
            print("lightFM recommendation")
            return self.lightFmRecommender.recommend(customer_id, n)
        if self.similarityRecommender.can_recommend(customer_id):
            print("similarity recommendation")
            return self.similarityRecommender.recommend(customer_id, n)
        else:
            print("pop recommendation")
            return self.popRecommender.recommend(n)