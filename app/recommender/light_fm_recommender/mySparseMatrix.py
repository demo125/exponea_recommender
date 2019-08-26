from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np

class MySparseMatrix:
    
    #df contains rows - customer_id - product_id - number of customer views of product
    def __init__(self, df):
        self.unique_customers = np.unique(np.sort(df.customer_id))
        self.unique_products = np.unique(np.sort(df.product_id))
        self.rating = list(df.rating)
        self.n = len(self.unique_customers)
        self.m = len(self.unique_products)
        
        self.customer_idxs = np.arange(self.n)
        self.customer_id_to_idx_map = dict((id, idx) for id, idx in zip(self.unique_customers, self.customer_idxs))
        self.customer_idx_to_id_map = dict((idx, id) for id, idx in zip(self.unique_customers, self.customer_idxs))
        
        self.product_idxs = np.arange(self.m)
        self.product_id_to_idx_map = dict((id, idx) for id, idx in zip(self.unique_products, self.product_idxs))
        self.product_idx_to_id_map = dict((idx, id) for id, idx in zip(self.unique_products, self.product_idxs))
        
        rows = df.customer_id.astype('category', self.unique_customers).cat.codes 
        cols = df.product_id.astype('category', self.unique_products).cat.codes 
        self.sparse = csr_matrix((self.rating, (rows, cols)), shape=(self.n, self.m))
        
    
    def get_matrix(self):
        return self.sparse
        
    def customer_id_to_idx(self, id):
        try:
            return self.customer_id_to_idx_map[id]
        except:
            print("Invalid customer id " + str(id))
    
    def customer_idx_to_id(self, idx):
        try:
            return int(self.customer_idx_to_id_map[idx])
        except:
            print("Invalid customer idx " + str(idx))
    
    
    def product_id_to_idx(self, id):
        try:
            return self.product_id_to_idx_map[id]
        except:
            print("Invalid product id " + str(id))
    
    def product_idx_to_id(self, idx):
        try:
            return int(self.product_idx_to_id_map[idx])
        except:
            print("Invalid prduct idx " + str(idx))
            
    def get_customer_products(self, customer_id):
        products = []
        for i, x in enumerate(np.array(self.sparse[self.customer_id_to_idx(customer_id)].todense()).flatten()):
            if x != 0:
                products.append(self.product_idx_to_id(i))
        return np.array(products, dtype=np.int)
    