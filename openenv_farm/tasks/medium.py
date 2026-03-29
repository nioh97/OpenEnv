"""Medium task: tighter resources, profit-focused framing."""


def get_initial_conditions():
    return {
        "soil_moisture": 0.35,
        "soil_nutrients": {"N": 0.35, "P": 0.35, "K": 0.35},
        "pest_level": 0.12,
        "water_available": 6000,
        "budget": 1700,
    }
