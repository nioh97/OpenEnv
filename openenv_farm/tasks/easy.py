"""Easy task: favorable starting state oriented toward yield."""


def get_initial_conditions():
    return {
        "soil_moisture": 0.6,
        "soil_nutrients": {"N": 0.7, "P": 0.7, "K": 0.7},
        "pest_level": 0.03,
        "water_available": 12000,
        "budget": 3500,
    }
