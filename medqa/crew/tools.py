from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

class ConsensusTool:
    def __init__(self, threshold: float = 0.85):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = threshold

    def check_consensus(self, answers: List[str]) -> bool:
        embeddings = self.model.encode(answers)
        similarity_matrix = cosine_similarity(embeddings)
        print(similarity_matrix)
        return np.all(similarity_matrix >= self.threshold)