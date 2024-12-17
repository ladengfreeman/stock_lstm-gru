from fastapi import FastAPI
from pydantic import BaseModel
from predictmodel import predict

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Hello, world"}


class StockRequest(BaseModel):
    stock_name: str
    start_date: str
    end_date: str


@app.post('/predict')
def predict_stock(request: StockRequest):
    predicted_prices = predict(request.stock_name, request.start_date, request.end_date)
    return {
        "predicted_prices": {
            "Day 1": round(predicted_prices[0], 2),
            "Day 2": round(predicted_prices[1], 2),
            "Day 3": round(predicted_prices[2], 2)
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
