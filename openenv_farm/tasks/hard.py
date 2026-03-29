"""Hard task: stressed field and scarce inputs for sustainability-style play."""


def get_initial_conditions():
    return {
        "soil_moisture": 0.35,
        "soil_nutrients": {"N": 0.3, "P": 0.3, "K": 0.3},
        "pest_level": 0.15,
        "water_available": 5000,
        "budget": 1500,
    }
