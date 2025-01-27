import numpy as np
import pandas as pd

def calculate_psi(base, current, bins=10):
    """
    Функция для расчета Population Stability Index (PSI)
    
    :param base: Исходная выборка (например, обучающие данные)
    :param current: Текущая выборка (например, тестовые данные)
    :param bins: Количество интервалов для разбиения данных (по умолчанию 10)
    :return: Значение PSI
    """
    # Разбиваем данные на бины
    base_percents = np.histogram(base, bins=bins, density=True)[0]
    current_percents = np.histogram(current, bins=bins, density=True)[0]
    
    # Применяем логарифм и суммируем по всем бинам
    psi = np.sum((base_percents - current_percents) * np.log(base_percents / current_percents), where=(base_percents != 0) & (current_percents != 0))
    
    return psi

# Пример использования
base_data = np.random.normal(0, 1, 1000)  # Исходная выборка (например, обучающие данные)
current_data = np.random.normal(0.5, 1, 1000)  # Текущая выборка (например, тестовые данные)

psi_value = calculate_psi(base_data, current_data)
print(f"PSI value: {psi_value}")
