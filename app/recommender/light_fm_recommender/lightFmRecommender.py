from lightfm import LightFM
import numpy as np
import pandas as pd
import os
from lightfm.data import Dataset
from recommender.recommenderABC import RecommenderABC

#preprocessing and training logic is described in notebooks/1_collaborative_recommender.ipynb
class LightFmRecommender(RecommenderABC):

    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = dir_path + '/../data/'
        self.model = None

        self.min_product_events_count = 50
        self.min_customer_events_count = 7

    def train(self):
        events = pd.read_csv(self.data_dir + "/dataset_events.csv")

        only_views = events[events.type=="view_product"].drop(columns=['type', 'timestamp'])
        grouped_views_by_customer_id = only_views.groupby(['customer_id', 'product_id']).product_id.agg('count').to_frame('n_views').reset_index()

        only_cart = events[events.type=="add_to_cart"].drop(columns=['type', 'timestamp'])
        grouped_cart_by_customer_id = only_cart.groupby(['customer_id', 'product_id']).product_id.agg('count').to_frame('n_cart').reset_index()

        only_purchases = events[events.type=="purchase_item"].drop(columns=['type', 'timestamp'])
        grouped_purchases_by_customer_id = only_purchases.groupby(['customer_id', 'product_id']).product_id.agg('count').to_frame('n_purchases').reset_index()

        customer_product = events[['customer_id', 'product_id']].drop_duplicates().reset_index(drop=True)
        event_counts = pd.merge(customer_product, grouped_views_by_customer_id,  how='left', left_on=['customer_id','product_id'], right_on = ['customer_id','product_id'])
        event_counts['n_views'].fillna(0, inplace=True)
        event_counts = pd.merge(event_counts, grouped_cart_by_customer_id,  how='left', left_on=['customer_id','product_id'], right_on = ['customer_id','product_id'])
        event_counts['n_cart'].fillna(0, inplace=True)
        event_counts = pd.merge(event_counts, grouped_purchases_by_customer_id,  how='left', left_on=['customer_id','product_id'], right_on = ['customer_id','product_id'])
        event_counts['n_purchases'].fillna(0, inplace=True)

        event_counts['rating'] = 0
        idxs = event_counts[event_counts.n_views == 1].index
        event_counts.loc[idxs, 'rating'] = 1

        idxs = event_counts[event_counts.n_views > 1].index
        event_counts.loc[idxs, 'rating'] = 2

        idxs = event_counts[event_counts.n_cart == 1].index
        event_counts.loc[idxs, 'rating'] = 3

        idxs = event_counts[event_counts.n_cart > 1].index
        event_counts.loc[idxs, 'rating'] = 4

        idxs = event_counts[event_counts.n_purchases == 1].index
        event_counts.loc[idxs, 'rating'] = 5

        idxs = event_counts[event_counts.n_purchases > 1].index
        event_counts.loc[idxs, 'rating'] = 6

        
        grouped_events_by_product_id = events.groupby(['product_id']).customer_id.agg('count').to_frame('n_events').reset_index()
        products_with_more_than_n_events = grouped_events_by_product_id[grouped_events_by_product_id.n_events>self.min_product_events_count]
        products_with_more_than_n_events = np.array(products_with_more_than_n_events.product_id)

        
        grouped_events_by_customer_id = events.groupby(['customer_id']).product_id.agg('count').to_frame('n_events').reset_index()
        customers_with_more_than_n_events = grouped_events_by_customer_id[grouped_events_by_customer_id.n_events>self.min_customer_events_count]
        customers_with_more_than_n_events = np.array(customers_with_more_than_n_events.customer_id)


        filtered_event_counts = event_counts.loc[event_counts.product_id.isin(products_with_more_than_n_events)]
        filtered_event_counts = filtered_event_counts.loc[event_counts.customer_id.isin(customers_with_more_than_n_events)]

        ratings =  filtered_event_counts.drop(columns=['n_views','n_cart','n_purchases']).reset_index(drop=True).values

        dataset = Dataset()
        dataset.fit(
            users = set(ratings[:, 0]),
            items = set(ratings[:, 1])
        )
        _, interactions = dataset.build_interactions(ratings)

        mappings = dataset.mapping()
        self.user_mapping, self.item_mapping = mappings[0], mappings[2]
        self.product_idxs = np.array(list(self.item_mapping.values()))
        self.product_ids = np.array(list(self.item_mapping.keys()))

        self.model = LightFM(loss='warp', no_components=40)

        self.model.fit_partial(interactions, epochs=15, verbose=True)

        print('LightFM model trained')

    def can_recommend(self , customer_id):
        if self.model is None: self.train()
        return customer_id in self.user_mapping
        
    def recommend(self, customer_id, n):
        if self.model is None: self.train()

        customer_idx = self.user_mapping[customer_id]

        scores = self.model.predict([customer_idx], self.product_idxs)
        sorted_idxs = np.argsort(-scores)[:n]

        return sorted_idxs.tolist()
        