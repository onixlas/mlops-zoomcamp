from datetime import datetime

import pandas as pd

import batch


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

import batch

data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

columns = ['PULocationID', 'DOLocationID', 
'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df = pd.DataFrame(data, columns=columns)

input_file = batch.get_input_path(2023, 1)

options = {
    'client_kwargs': {
        'endpoint_url': 'http://localhost:4566'
        }
    }


df.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

