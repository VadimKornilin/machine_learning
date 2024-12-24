from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Загружаем модель
model = joblib.load('model.joblib')

# Создаем экземпляр FastAPI
app = FastAPI()

# Модель для валидации входных данных
class InputData(BaseModel):
    feature: float

# Определяем маршрут для предсказания
@app.post("/predict/")
async def predict(data: InputData):
    # Преобразуем данные во вход для модели
    input_data = np.array([[data.feature]])
    
    # Получаем предсказание от модели
    prediction = model.predict(input_data)[0]
    
    return {"prediction": prediction}

# Стартовый маршрут
@app.get("/")
async def read_root():
    return {"message": "Welcome to the ML model API!"}

