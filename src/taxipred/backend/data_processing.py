from taxipred.utils.constants import CLEAN_TAXI_CSV_PATH
import pandas as pd
import json


class CleanTaxiData:
    def __init__(self):
        self.df = pd.read_csv(CLEAN_TAXI_CSV_PATH)

    def to_json(self):
        return json.loads(self.df.to_json(orient = "records"))
