from sklearn.metrics import roc_auc_score

def custom_gini(y_true, y_pred):
    """
    Кастомная метрика: 2 * ROC AUC - 1
    """
    roc_auc = roc_auc_score(y_true, y_pred)
    gini = 2 * roc_auc - 1
    return gini

# Обёртка для CatBoost
def catboost_custom_metric(y_true, y_pred):
    """
    Кастомная метрика для CatBoost в формате (имя метрики, значение, направление оптимизации)
    """
    gini = custom_gini(y_true, y_pred)
    return "Custom_Gini", gini, True  # True означает, что метрика должна максимизироваться