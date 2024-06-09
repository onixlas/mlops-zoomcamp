import requests
from io import BytesIO
from typing import List

import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    df = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')

    return df