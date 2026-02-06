from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class BaseModel(ABC):
    """
    Interface pour tout modèle de prédiction de défauts Terraform.
    """

    @abstractmethod
    def predict(self, vectors: Dict[str, List[float]]) -> Dict[str, int]:
        """
        Prédit si chaque bloc est fault-prone ou non.

        Args:
            vectors (Dict[str, List[float]]): Dictionnaire {block_id: vecteur de caractéristiques}

        Returns:
            Dict[str, int]: Dictionnaire {block_id: 0 ou 1}, 1 si defectueux
        """
        pass

    def predict_with_confidence(
        self, vectors: Dict[str, List[float]]
    ) -> Dict[str, Tuple[int, float]]:
        """
        Optionnel : Retourne aussi le score de confiance (probabilité).
        Par défaut : utilise predict() et fixe la confiance à 1.0.
        """
        return {
            block_id: (label, 1.0) for block_id, label in self.predict(vectors).items()
        }

    @abstractmethod
    def describe(self) -> str:
        """
        Donne une brève description du modèle.
        """
        pass
