import pandas as pd
import numpy as np
from recommender.similarity_recommender.similarityABC import SimilarityABC
from sklearn.preprocessing import OneHotEncoder

#preprocessing and training logic is described in notebooks/2_similarity_recommender.ipynb
class GenderSimilarity(SimilarityABC):

    def __init__(self, weight=1):
        super().__init__(weight=weight, name="GenderSimilarity")

    def do_fetch_data(self):
        data = pd.read_csv(self.data_dir + "/dataset_catalog.csv")
        return data.drop(columns=['category_id', 'category_path', 'brand', 'description', 'price'])

    def do_preprocess(self):

        self.data['Man'] = 0
        self.data['Woman'] = 0
        self.data['Child'] = 0
        self.data['Unisex'] = 0
        self.data['Other'] = 0

        indexes = self.data[self.data.gender == 'Man'].index
        self.data.loc[indexes, 'Man'] = 1

        indexes = self.data[self.data.gender == 'Woman'].index
        self.data.loc[indexes, 'Woman'] = 1

        indexes = self.data[self.data.gender == 'Other'].index
        self.data.loc[indexes, 'Other'] = 1
        self.data.loc[indexes, 'Man'] = 1
        self.data.loc[indexes, 'Woman'] = 1
        self.data.loc[indexes, 'Unisex'] = 1

        indexes = self.data[self.data.gender == 'Child'].index
        self.data.loc[indexes, 'Child'] = 1

        indexes = self.data[self.data.gender == 'Unisex'].index
        self.data.loc[indexes, 'Unisex'] = 1
        self.data.loc[indexes, 'Man'] = 1
        self.data.loc[indexes, 'Woman'] = 1

        return self.data[['Man', 'Woman', 'Child', 'Unisex', 'Other']].values

    def do_train(self):
        from sklearn.neighbors import NearestNeighbors

        neigh = NearestNeighbors(n_neighbors=5, metric='cosine')
            
        neigh.fit(self.preprocessed_data)
        return neigh
        
    def do_get_product_similarities(self, product_id):
        product_idx = product_id - 1
        product_count = self.preprocessed_data.shape[0] 

        d = [self.preprocessed_data[product_idx]]
        distances, neighbours = self.model.kneighbors(d, product_count, return_distance=True)

        distances = distances.flatten()
        neighbours = neighbours.flatten()
        similarity = np.ones(distances.shape) - distances
        all_product_similarities = np.zeros(product_count)
        all_product_similarities[neighbours] = similarity
        
        return all_product_similarities