# TODO: implement similarities computation
from dataclasses import dataclass

import numpy as np


def cosine_similarity(cav1, cav2):
	dot_product = np.dot(cav1, cav2)
	norm_cav1 = np.linalg.norm(cav1)
	norm_cav2 = np.linalg.norm(cav2)

	if norm_cav1 == 0 or norm_cav2 == 0:
		return 0.0

	similarity = dot_product / (norm_cav1 * norm_cav2)
	return similarity

@dataclass
class SimilaritiesConfig:
	method: str = "default"

class Similarities:
	def __init__(self, cfg: SimilaritiesConfig = SimilaritiesConfig()):
		self.cfg = cfg

	def compute_similarity_scores(self, cav1, cav2):
		if self.cfg.method == "cosine" or self.cfg.method == "default":
			return cosine_similarity(cav1, cav2)
		if self.cfg.method == "dotproduct":
			return np.dot(cav1, cav2)
		if self.cfg.method == "euclidean":
			return np.linalg.norm(cav1 - cav2)
		else:
			print("Method not recognized, using default (cosine similarity).")
			return cosine_similarity(cav1, cav2)
