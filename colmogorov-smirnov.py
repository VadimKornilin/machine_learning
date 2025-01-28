import numpy as np
from scipy.stats import ks_2samp

# Предсказанные вероятности модели
predictions = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.6])
true_labels = np.array([0, 0, 1, 1, 1, 0, 1, 1])

# Разделение предсказаний на классы
p_1 = predictions[true_labels == 1]  # Предсказания для y=1
p_0 = predictions[true_labels == 0]  # Предсказания для y=0

# Вычисление статистики Колмогорова-Смирнова
ks_stat, p_value = ks_2samp(p_1, p_0)

print(f"KS-статистика: {ks_stat}")
print(f"P-значение: {p_value}")

# Интерпретация
if p_value < 0.05:
    print("Модель обладает значимой дискриминационной способностью.")
else:
    print("Дискриминационная способность модели незначима.")
