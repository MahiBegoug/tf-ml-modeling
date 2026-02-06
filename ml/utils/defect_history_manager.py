import json
import os
from datetime import datetime
from typing import Dict

from app import config
from infrastructure.git.git_adapter import get_latest_commit_hash


def load_defect_history() -> Dict[str, list]:
    """
    Charge l'historique des prédictions par bloc. Si le fichier n'existe pas, retourne un dict vide.
    """
    if not os.path.exists(config.DEFECT_HISTORY_PATH):
        return {}
    with open(config.DEFECT_HISTORY_PATH, "r") as f:
        return json.load(f)


def save_defect_history(history: Dict[str, list]):
    """
    Sauvegarde l'historique dans defect_history.json.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(config.DEFECT_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)


def update_defect_history(predictions: Dict[str, int]):
    """
    Met à jour l'historique des défauts avec la prédiction du modèle pour chaque bloc.
    On enregistre aussi le commit et la date de prédiction.
    """
    history = load_defect_history()
    current_commit = get_latest_commit_hash()
    now = datetime.now().isoformat()

    for block_id, pred in predictions.items():
        entry = {"commit": current_commit, "fault_prone": pred, "date": now}

        if block_id not in history:
            history[block_id] = [entry]
        else:
            # Ne pas ajouter deux fois une prédiction pour le même commit
            if not any(p["commit"] == current_commit for p in history[block_id]):
                history[block_id].append(entry)

    save_defect_history(history)
