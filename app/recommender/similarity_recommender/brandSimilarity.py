import pandas as pd
import numpy as np
from recommender.similarity_recommender.similarityABC import SimilarityABC
from sklearn.preprocessing import OneHotEncoder

class BrandSimilarity(SimilarityABC):

    def __init__(self, weight=1):
        super().__init__(weight=weight, name="BrandSimilarity")

    def do_fetch_data(self):
        data = pd.read_csv(self.data_dir + "/dataset_catalog.csv")
        return data.drop(columns=['category_id', 'category_path', 'gender', 'description', 'price'])

    def do_preprocess(self):
        brands = self.data.brand.values

        OHE = OneHotEncoder()
        ohe_mat = OHE.fit_transform(brands.reshape(len(brands), 1))

        return ohe_mat

    def do_train(self):
        return self.preprocessed_data.dot(self.preprocessed_data.T)

    def do_get_product_similarities(self, product_id):
        product_idx = product_id - 1
        return np.array(self.model[product_idx].todense()).flatten()

