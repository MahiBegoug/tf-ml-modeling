import random
from typing import Dict, List, Tuple

from ml.models.base_model import BaseModel


class DummyModel(BaseModel):
    """
    Modèle fictif qui prédit 0 ou 1 au hasard pour chaque bloc.
    Utile pour tester l'intégration de la prédiction.
    """

    def predict(self, vectors: Dict[str, List[float]]) -> Dict[str, int]:
        predictions = {}
        for block_id in vectors:
            predictions[block_id] = random.choice([0, 1])
        return predictions

    def predict_with_confidence(
        self, vectors: Dict[str, List[float]]
    ) -> Dict[str, Tuple[int, float]]:
        return {
            block_id: (random.randint(0, 1), round(random.uniform(0.5, 1.0), 2))
            for block_id in vectors
        }

    def describe(self) -> str:
        return "🧠 Modèle fictif (Dummy) pour tests uniquement."
