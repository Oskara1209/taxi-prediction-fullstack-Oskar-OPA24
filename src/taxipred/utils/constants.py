from importlib.resources import files

TAXI_CSV_PATH = files("taxipred").joinpath("data/taxi_trip_pricing.csv")

CLEAN_TAXI_CSV_PATH = files("taxipred").joinpath("data/taxi_trip_pricing_clean.csv")

# DATA_PATH = Path(__file__).parents[1] / "data"