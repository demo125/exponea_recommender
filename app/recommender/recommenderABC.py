from abc import ABC, abstractmethod

class RecommenderABC(ABC):
    
    @abstractmethod
    def train(self):
        pass
    
    #find out if recommender model can perform recommendation
    @abstractmethod
    def can_recommend(self, customer_id):
        pass
    
    @abstractmethod
    def recommend(self, customer_id, n):
        pass
