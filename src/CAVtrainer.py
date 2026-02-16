from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
import os

def compute_cav(concept_acts, random_acts):

    X = np.concatenate([concept_acts, random_acts], axis=0)

    y = np.concatenate([
        np.ones(len(concept_acts)),
        np.zeros(len(random_acts))
    ])

    clf = LogisticRegression(max_iter=3000, solver="liblinear")
    clf.fit(X, y)

    cav_vector = clf.coef_[0]

    # Normalizzazione opzionale
    cav_vector = cav_vector / np.linalg.norm(cav_vector)

    return cav_vector


def save_cav(cav_vector, concept_name, layer_name, output_path):

    os.makedirs(output_path, exist_ok=True)

    safe_layer_name = layer_name.replace(".", "_")

    filename = f"{concept_name}_{safe_layer_name}_cav.npy"
    filepath = os.path.join(output_path, filename)

    np.save(filepath, cav_vector)

@dataclass
class CAVConfig:
    concept_name: str
    layer_name: str
    output_path: str

class CAVtrainer:
    def __init__(self, model, output_path):
        self.model = model
        self.output_path = output_path

    def train_cav(self, concept_acts, random_acts, config: CAVConfig):
        cav_vector = compute_cav(concept_acts, random_acts)
        # save_cav(cav_vector, config.concept_name, config.layer_name, self.output_path)
        return cav_vector
