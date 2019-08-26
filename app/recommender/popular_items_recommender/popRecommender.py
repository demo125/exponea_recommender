import pandas as pd
import numpy as np
import os
from recommender.recommenderABC import RecommenderABC

class PopRecommender(RecommenderABC):

    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = dir_path + '/../data/'
        self.sorted_popular_item_ids = None

    def train(self):
        events = pd.read_csv(self.data_dir + "/dataset_events.csv")

        purchases = events[events.type == 'purchase_item'].drop(columns=['timestamp'])
        purchases_with_no_duplicates = purchases.drop_duplicates(['product_id', 'customer_id'])

        grouped_purchases_count_by_product_id = purchases_with_no_duplicates.groupby(['product_id']).customer_id.agg('count').to_frame('n_unique_purchases').reset_index()

        sorted_popular_items = grouped_purchases_count_by_product_id.sort_values('n_unique_purchases', ascending=False)
        
        self.sorted_popular_item_ids = sorted_popular_items.product_id.values
        print('pop recommender trained')
    
    def can_recommend(self):
        return True

    def recommend(self, n):
        if self.sorted_popular_item_ids is None: self.train()
        print('pop recommendation')
        return self.sorted_popular_item_ids[:n].tolist()

