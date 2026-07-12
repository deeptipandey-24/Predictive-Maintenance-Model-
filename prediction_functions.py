def engineer_features(raw_input: dict) -> dict:
    """
    Takes raw sensor readings a user would realistically provide and computes
    all derived features the model was trained on, in correct dependency order.

    Required raw inputs:
        - Type
        - Air temperature [K]
        - Process temperature [K]
        - Rotational speed [rpm]
        - Torque [Nm]
        - Tool wear [min]
    """
    data = raw_input.copy()

    required_raw = [
        'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    ]
    missing = [c for c in required_raw if c not in data]
    if missing:
        raise ValueError(f"Missing required raw inputs: {missing}")

    # ── Step 1: base derived features ──
    data['Heat_Strain'] = data['Process temperature [K]'] - data['Air temperature [K]']
    data['Power'] = data['Torque [Nm]'] * data['Rotational speed [rpm]']

    # ── Step 2: features that depend on Step 1 ──
    data['Thermal_Stress_Ratio'] = data['Power'] / (data['Heat_Strain'] + 1)
    data['High_Power_Hazard'] = int(data['Power'] > 86000)
    data['Underpower_Stall_Risk'] = int(data['Power'] < 30000)

    # ── Step 3: independent engineered features ──
    data['Tool_Wear_Squared'] = data['Tool wear [min]'] ** 2
    data['Torque_Wear_Impact'] = data['Torque [Nm]'] * data['Tool wear [min]']

    return data


def predict_machine_failure_risk(raw_user_input: dict, model, config):
    # Step 1: compute all engineered features internally
    full_data = engineer_features(raw_user_input)

    # Step 2: validate against what the pipeline actually expects
    expected_cols = config['feature_columns']
    missing = set(expected_cols) - set(full_data.keys())
    if missing:
        raise ValueError(f"Missing required fields after feature engineering: {missing}")

    df = pd.DataFrame([full_data])[expected_cols]

    if df.isnull().any().any():
        raise ValueError("Input contains null/missing values")

    # Step 3: predict
    failure_probability = model.predict_proba(df)[0, 1]
    failure_percentage = round(failure_probability * 100, 1)

    threshold = config['threshold']
    if failure_probability >= 0.70:
        risk_level = "Critical"
    elif failure_probability >= threshold:
        risk_level = "High"
    elif failure_probability >= threshold * 0.6:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    return {
        'failure_probability_pct': failure_percentage,
        'risk_level': risk_level,
        'recommended_action': (
            "Immediate inspection required" if risk_level == "Critical" else
            "Schedule maintenance soon" if risk_level == "High" else
            "Monitor closely" if risk_level == "Moderate" else
            "No action needed"
        )
    }
