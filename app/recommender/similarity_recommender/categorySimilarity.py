import pandas as pd
import numpy as np
from recommender.similarity_recommender.similarityABC import SimilarityABC
from sklearn.neighbors import NearestNeighbors

#preprocessing and training logic is described in notebooks/2_similarity_recommender.ipynb
class CategorySimilarity(SimilarityABC):

    def __init__(self, weight=1):
        super().__init__(weight=weight, name="CategorySimilarity")

    def do_fetch_data(self):
        data = pd.read_csv(self.data_dir + "/dataset_catalog.csv")
        return data.drop(columns=['brand', 'gender', 'description', 'price'])

    def do_preprocess(self):
        category_names = set()
        categories = {}

        category_paths = self.data[['category_id', 'category_path']].drop_duplicates()
        category_path_part_names = set()
        for i, r in category_paths.iterrows():
            for cat in r.category_path.split('|'):
                if cat not in category_paths:
                    category_paths[cat] = 0
                    category_path_part_names.add(cat)
                category_paths.loc[i, cat] = 1

        products_with_column_categories = pd.merge(self.data, category_paths,  how='left', left_on=['category_id'], right_on = ['category_id'])
        products_with_column_categories = products_with_column_categories.drop(columns=['product_id', 'category_id', 'category_path_x', 'category_path_y'])

        return products_with_column_categories.values

    def do_train(self):
        neigh = NearestNeighbors(n_neighbors=5, metric='cosine')
        neigh.fit(self.preprocessed_data)

        return neigh
    
    def do_get_product_similarities(self, product_id):

        product_count = self.preprocessed_data.shape[0]
        product_idx = product_id - 1
        distances, neighbours = self.model.kneighbors([self.preprocessed_data[product_idx]], product_count, return_distance=True)
        distances = distances.flatten()
        neighbours = neighbours.flatten()
        similarity = np.ones(distances.shape) - distances
        all_product_similarities = np.zeros(product_count)
        all_product_similarities[neighbours] = similarity
        return all_product_similarities
