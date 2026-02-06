import os
from typing import Dict, List, Tuple

import joblib

from app import config
from ml.models.base_model import BaseModel


class RandomForestModel(BaseModel):
    """
    Modèle Random Forest entraîné pour détecter les blocs Terraform fautifs.
    Utilise un scaler et un classifieur entraînés, chargés depuis un fichier .joblib.
    Ce fichier doit contenir un dictionnaire avec les clés : {"model": ..., "scaler": ...}
    """

    def __init__(self, model_path: str = config.RF_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle non trouvé : {model_path}")

        bundle = joblib.load(model_path)
        self.model = bundle.get("model")
        self.scaler = bundle.get("scaler")

        if self.model is None or self.scaler is None:
            raise ValueError(
                "Le fichier joblib doit contenir les clés 'model' et 'scaler'"
            )

    def predict(self, vectors: Dict[str, List[float]]) -> Dict[str, int]:
        """
        Prédit le label de chaque bloc Terraform après application du scaler.
        Retourne un dictionnaire {block_id: 0|1}.
        """
        block_ids = list(vectors.keys())
        X = list(vectors.values())

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        return {block_id: int(pred) for block_id, pred in zip(block_ids, predictions)}

    def predict_with_confidence(
        self, vectors: Dict[str, List[float]]
    ) -> Dict[str, Tuple[int, float]]:
        """
        Prédit le label de chaque bloc Terraform ainsi que son score de confiance (proba).
        Retourne un dictionnaire {block_id: (label, proba)}.
        """
        block_ids = list(vectors.keys())
        X = list(vectors.values())

        X_scaled = self.scaler.transform(X)

        predicted_probs = self.model.predict_proba(X_scaled)
        results = {}

        for i, block_id in enumerate(block_ids):
            proba = predicted_probs[i][1]
            label = int(proba >= 0.5)
            results[block_id] = (label, round(proba, 4))

        return results

    def describe(self) -> str:
        """
        Retourne une description du modèle et du scaler utilisés.
        Le nombre de features est déterminé dynamiquement.
        """
        model_type = type(self.model).__name__
        scaler_type = type(self.scaler).__name__

        if hasattr(self.model, "n_features_in_"):
            num_features = self.model.n_features_in_
        elif hasattr(self.scaler, "scale_"):
            num_features = len(self.scaler.scale_)
        else:
            num_features = "?"

        return f"🧠 Modèle : {model_type} | 🔧 Scaler : {scaler_type} | 🔁 Features : {num_features}"
