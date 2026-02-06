from app import config
from ml.models.base_model import BaseModel
from ml.models.dummy_model import DummyModel
from ml.models.random_forest_model import RandomForestModel
from ml.models.sklearn_model import SklearnModel


class ModelFactory:
    @staticmethod
    def get_model(model_type: str) -> BaseModel:
        """
        Retourne une instance de modèle prédictif selon le type spécifié.

        Args:
            model_type (str): Type du modèle (ex: 'dummy', 'randomforest', 'lightgbm', etc.)

        Returns:
            BaseModel: instance du modèle
        """
        model_type = model_type.lower()

        if model_type == "dummy":
            return DummyModel()

        if model_type == "randomforest":
            return RandomForestModel()

        supported_models = {
            "lightgbm": config.LIGHTGBM_MODEL_PATH,
            "logisticreg": config.LOGISTICREG_MODEL_PATH,
            "naivebayes": config.NAIVEBAYES_MODEL_PATH,
        }

        if model_type in supported_models:
            return SklearnModel(model_type, supported_models[model_type])

        raise ValueError(f"Modèle non supporté : {model_type}")
