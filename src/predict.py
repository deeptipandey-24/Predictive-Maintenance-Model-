
import pandas as pd
import joblib
import json
from feature_engineering import engineer_features
from logging_utils import setup_logger, log_prediction

def load_production_model(models_dir: str):
    model = joblib.load(f'{models_dir}/voting_ensemble_final.joblib')
    with open(f'{models_dir}/model_config.json') as f:
        config = json.load(f)
    return model, config

def predict_machine_failure_risk(raw_user_input: dict, model, config, logger=None, machine_id=None):
    try:
        full_data = engineer_features(raw_user_input)
        expected_cols = config['feature_columns']
        missing = set(expected_cols) - set(full_data.keys())
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        df = pd.DataFrame([full_data])[expected_cols]
        if df.isnull().any().any():
            raise ValueError("Input contains null/missing values")

        proba = model.predict_proba(df)[0, 1]
        pct = round(proba * 100, 1)
        threshold = config['threshold']

        if proba >= 0.70:
            risk_level = "Critical"
        elif proba >= threshold:
            risk_level = "High"
        elif proba >= threshold * 0.6:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        result = {
            'failure_probability_pct': pct,
            'risk_level': risk_level,
            'recommended_action': (
                "Immediate inspection required" if risk_level == "Critical" else
                "Schedule maintenance soon" if risk_level == "High" else
                "Monitor closely" if risk_level == "Moderate" else
                "No action needed"
            )
        }

        if logger:
            log_prediction(logger, machine_id, 'success', raw_user_input, result=result)
        return result

    except Exception as e:
        if logger:
            log_prediction(logger, machine_id, 'error', raw_user_input, error=str(e))
        raise
