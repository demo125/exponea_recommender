from abc import ABC, abstractmethod

class RecommenderABC(ABC):
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def can_recommend(self):
        pass
    
    @abstractmethod
    def recommend(self):
        pass
