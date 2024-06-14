import os
import numpy as np
import joblib
import dill
import json
from datetime import datetime

def save_dict_to_json(data, filename_prefix):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(data, f)

    print(f"Dictionary saved as '{filename}'")

def load_function(filename):
    with open(filename, 'rb') as f:
        loaded_function = dill.load(f)
    return loaded_function

def load_models(model_dir):
    model_paths = [os.path.join(model_dir, fname) for fname in os.listdir(model_dir) if fname.endswith('.pkl')]
    models = [joblib.load(path) for path in model_paths]
    return models

def base_model(models, X_unseen):
    all_predictions = np.zeros((len(X_unseen), len(models), 2))

    for i, model in enumerate(models):
        y_pred_proba = model.predict_proba(X_unseen)
        all_predictions[:, i, :] = y_pred_proba

    final_predictions = np.mean(all_predictions, axis=1)
    return final_predictions[:, 0], final_predictions[:, 1]