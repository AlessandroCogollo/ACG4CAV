# TODO: implement similarities computation
from dataclasses import dataclass


@dataclass
class SimilaritiesConfig:
	method: str = "default"

class Similarities:
	def __init__(self, cfg: SimilaritiesConfig = SimilaritiesConfig()):
		self.cfg = cfg

	def compute_similarity_scores(self, cav_vector, cav_vector1):
		pass


def main():
	print("Computing similarities between concepts...")

if __name__ == "__main__":
	main()
