import pandas as pd
import numpy as np
from recommender.similarity_recommender.similarityABC import SimilarityABC
from sklearn.neighbors import NearestNeighbors

#preprocessing and training logic is described in notebooks/2_similarity_recommender.ipynb
class PriceSimilarity(SimilarityABC):

    def __init__(self, weight=1):
        super().__init__(weight=weight, name="PriceSimilarity")

    def do_fetch_data(self):
        data = pd.read_csv(self.data_dir + "/dataset_catalog.csv")
        return data.drop(columns=['category_id', 'category_path', 'brand', 'gender', 'description'])

    def do_preprocess(self):
        return self.data.price.values.reshape(-1, 1)

    def do_train(self):
        neigh = NearestNeighbors(n_neighbors=5, metric='euclidean')
            
        neigh.fit(self.preprocessed_data)

        return neigh

    def do_get_product_similarities(self, product_id):

        product_count = self.preprocessed_data.shape[0] 
        product_idx = product_id - 1
        distances, neighbours = self.model.kneighbors([self.preprocessed_data[product_idx]], product_count, return_distance=True)
        distances = distances.flatten()
        neighbours = neighbours.flatten()
        similarity = self.__distances_to_similarity(distances)
        
        all_product_similarities = np.zeros(product_count)
        all_product_similarities[neighbours] = similarity
        
        return all_product_similarities

    def __distances_to_similarity(self, distances):
        similarity = np.zeros(distances.shape)
        similarity[distances <= 3] = 1
        similarity[(distances > 3) & (distances <=5)] = 0.9
        similarity[(distances > 5) & (distances <=10)] = 0.8
        similarity[(distances > 10) & (distances <=25)] = 0.7
        similarity[(distances > 25) & (distances <=30)] = 0.6
        similarity[(distances > 30) & (distances <=40)] = 0.5
        similarity[(distances > 40) & (distances <=70)] = 0.4
        similarity[(distances > 70) & (distances <=100)] = 0.3
        similarity[(distances > 100) & (distances <=150)] = 0.2
        similarity[(distances > 150) & (distances <=250)] = 0.1
        return similarity