
def engineer_features(raw_input: dict) -> dict:
    """
    Takes raw sensor readings and computes all derived features
    the model was trained on, in correct dependency order.
    """
    data = raw_input.copy()

    required_raw = [
        'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    ]
    missing = [c for c in required_raw if c not in data]
    if missing:
        raise ValueError(f"Missing required raw inputs: {missing}")

    data['Heat_Strain'] = data['Process temperature [K]'] - data['Air temperature [K]']
    data['Power'] = data['Torque [Nm]'] * data['Rotational speed [rpm]']
    data['Thermal_Stress_Ratio'] = data['Power'] / (data['Heat_Strain'] + 1)
    data['High_Power_Hazard'] = int(data['Power'] > 86000)
    data['Underpower_Stall_Risk'] = int(data['Power'] < 30000)
    data['Tool_Wear_Squared'] = data['Tool wear [min]'] ** 2
    data['Torque_Wear_Impact'] = data['Torque [Nm]'] * data['Tool wear [min]']

    return data
