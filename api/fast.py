import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pytz
import joblib

from TaxiFareModel.params import PATH_TO_LOCAL_MODEL
from predict import download_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}


@app.get("/predict")
def predict(pickup_datetime, pickup_longitude, pickup_latitude,
          dropoff_longitude, dropoff_latitude, passenger_count):

    X_pred_dict = {
        'key': "2013-07-06 17:18:00.000000119",
        'pickup_datetime': pickup_datetime,
        'pickup_longitude': float(pickup_longitude),
        'pickup_latitude': float(pickup_latitude),
        'dropoff_longitude': float(dropoff_longitude),
        'dropoff_latitude': float(dropoff_latitude),
        'passenger_count': int(passenger_count)
    }

    X_pred = pd.DataFrame(X_pred_dict, index=[0])

    # create a datetime object from the user provided datetime
    pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")

    # localize the user datetime with NYC timezone
    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)

    # localize the datetime to UTC
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)
    formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

    X_pred['pickup_datetime'] = formatted_pickup_datetime

    pipeline = joblib.load(PATH_TO_LOCAL_MODEL)
    # pipeline = download_model(rm=False)

    y_pred = pipeline.predict(X_pred)

    X_pred_dict['prediction'] = y_pred[0]

    return X_pred_dict
