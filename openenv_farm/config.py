"""Central configuration for the farm simulation environment."""

from __future__ import annotations

# Episode horizon
MAX_DAYS = 90

# Initial resources
INITIAL_WATER = 8_000.0  # liters available for irrigation
INITIAL_BUDGET = 2_500.0  # currency units

# Soil & crop initialization
INITIAL_SOIL_MOISTURE = 0.46
INITIAL_SOIL_NUTRIENTS = {"N": 0.38, "P": 0.36, "K": 0.35}
INITIAL_CROP_HEALTH = 0.52
INITIAL_PEST_LEVEL = 0.06
INITIAL_GROWTH_PROGRESS = 0.08  # early sowing phase
INITIAL_CROP_STAGE = "sowing"

# Viability
CROP_DEATH_THRESHOLD = 0.05

# Agronomic targets (normalized nutrient indices 0–1)
OPTIMAL_MOISTURE_RANGE = (0.40, 0.70)
TARGET_NUTRIENT_RANGE = (0.34, 0.62)
NUTRIENT_CAP = 1.0  # hard clamp for soil storage model

# Stress / penalty thresholds
MOISTURE_STRESS_LOW = 0.22
MOISTURE_OVERWATER_THRESHOLD = 0.88
NUTRIENT_TOXIC_THRESHOLD = 0.90
STRESS_PEST_THRESHOLD = 0.55

# Irrigation & soil physics
IRRIGATION_EFFICIENCY = 0.82  # fraction of applied water retained as moisture gain
LITERS_TO_MOISTURE_SCALE = 8_500.0  # liters -> ~delta moisture for 1 ha conceptual field
SOIL_MOISTURE_DECAY_BASE = 0.038  # daily evapotranspiration-driven decay
TEMPERATURE_DECAY_FACTOR = 0.0022  # extra decay per °C above 22
RAIN_TO_MOISTURE = 0.00115  # mm precipitation -> moisture increment

# Fertilizer: kg applied -> bump to normalized nutrient pools (simple linear model)
FERTILIZER_YIELD_N = 0.022
FERTILIZER_YIELD_P = 0.020
FERTILIZER_YIELD_K = 0.019

# Pests
PEST_NATURAL_GROWTH = 0.045
PEST_MOISTURE_BOOST = 0.12  # pests like humid canopies (simplified)
PEST_TEMP_BOOST = 0.004  # per °C above 24
PESTICIDE_REDUCTION = 0.52
PEST_DAMAGE_TO_HEALTH = 0.11  # scale linking pest pressure to health loss

# Growth dynamics (logistic-style)
GROWTH_RATE_BASE = 0.095
GROWTH_LOGISTIC_K = 1.0  # carrying scale for nutrient/moisture composite
HEALTH_RECOVERY_RATE = 0.045
HEALTH_DAMAGE_RATE = 0.065

# Stage thresholds on growth_progress in [0, 1]
STAGE_THRESHOLDS = (
    ("sowing", 0.0),
    ("vegetative", 0.22),
    ("flowering", 0.50),
    ("mature", 0.78),
)

# Costs (currency)
COSTS = {
    "water_per_liter": 0.00085,
    "fertilizer_n_per_kg": 2.35,
    "fertilizer_p_per_kg": 2.05,
    "fertilizer_k_per_kg": 1.90,
    "pesticide_application": 48.0,
    "pesticide_externalities": 12.0,
}

# Reward shaping weights
REWARD_HEALTH_IMPROVEMENT = 2.4
REWARD_STAGE_PROGRESS = 6.0
REWARD_GROWTH_PROGRESS = 1.2
PENALTY_WATER_COST = 1.0
PENALTY_FERTILIZER_COST = 1.0
PENALTY_PESTICIDE_COST = 1.0
PENALTY_OVER_IRRIGATION = 3.5
PENALTY_OVER_FERTILIZATION = 4.0
PENALTY_STRESS = 1.8
REWARD_HARVEST_BASE = 55.0
REWARD_HARVEST_HEALTH_MULT = 42.0
REWARD_HARVEST_STAGE_MULT = 28.0
PENALTY_CROP_DEATH = 80.0

# Episode truncation (dense but small; avoids neutral time-outs)
TIME_LIMIT_PENALTY = 2.2

# Randomness namespace offset (weather vs other streams if extended)
WEATHER_SEED_OFFSET = 9_391_939
