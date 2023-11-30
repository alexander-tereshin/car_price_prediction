import pathlib

import pandas as pd

from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List

from preprocessing import CarPricePredictorPreprocessor

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


models_folder = pathlib.Path('.').resolve() / 'models'
preprocessor = CarPricePredictorPreprocessor(models_folder)


def predict_price(item: Item) -> float:
    """
    Predict the price for a single item.

    Args:
        item (Item): The input item.

    Returns:
        float: The predicted price.
    """
    processed_df = preprocessor.preprocess_data(pd.DataFrame([item.dict()]))
    return preprocessor.ridge_regressor.predict(processed_df)[0]
    # return processed_df


def predict_prices(items: Items) -> List[float]:
    """
    Predict prices for a list of items.

    Args:
        items (Items): The input items.

    Returns:
        List[float]: The predicted prices.
    """
    df = pd.DataFrame([item.dict() for item in items.objects])
    processed_df = preprocessor.preprocess_data(df)
    return preprocessor.ridge_regressor.predict(processed_df).tolist()


@app.get(path="/")
async def root():
    """
    Root endpoint that returns a greeting message.
    """
    return {"message": "Hello, User! This is The Car Price Prediction Web Service!"}


@app.post("/predict_item")
async def predict_item(item: Item) -> float:
    """
    Predict the price for a single item.

    Args:
        item (Item): The input item.

    Returns:
        float: The predicted price.
    """
    try:
        return predict_price(item)
        # return item
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_items")
async def predict_items(items: Items) -> List[float]:
    """
    Predict prices for a list of items.

    Args:
        items (Items): The input items.

    Returns:
        List[float]: The predicted prices.
    """
    try:
        return predict_prices(items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/predict_file")
async def predict_file(file: UploadFile) -> List[dict]:
    """
    Predict prices from a CSV file.

    Args:
        file (UploadFile): The input file.

    Returns:
        List[dict]: The predicted prices with input data.
    """
    try:
        df = pd.read_csv(file.file)
        processed_df = preprocessor.preprocess_data(df)
        predictions = preprocessor.ridge_regressor.predict(processed_df).tolist()

        # Combine input data with predictions
        result = []
        for _, row in df.iterrows():
            result.append({"input_data": row.to_dict(), "predicted_price": predictions.pop(0)})

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
