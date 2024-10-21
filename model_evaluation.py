"""Скрипт для оценки и тестирования модели."""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from data_preprocessing import preprocess_data
from feature_engineering import feature_engineering


def evaluate_model(model, X_test, y_test):
    # Предобработка тестовых данных
    X_test = preprocess_data(X_test, is_training=False)
    X_test = feature_engineering(X_test, is_training=False)

    # Загрузка имен признаков и обеспечение соответствия
    feature_names = joblib.load('models/feature_names.pkl')
    X_test = X_test[feature_names]

    predictions = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    # Проверка наличия обоих классов перед вычислением AUC
    unique_classes = len(set(y_test))
    if unique_classes == 2:
        auc = roc_auc_score(y_test, proba)
    else:
        print(
            f"Предупреждение: в y_test присутствует только {unique_classes} класс. ROC AUC не может быть вычислен."
        )
        auc = None

    report = classification_report(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    # Визуализация матрицы ошибок
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    return accuracy, auc, report, conf_matrix


if __name__ == "__main__":
    # Загрузка тестовых данных
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')['Churn']

    # Загрузка модели
    model = joblib.load('models/churn_model.pkl')

    # Оценка модели
    accuracy, auc, report, conf_matrix = evaluate_model(model, X_test, y_test)

    # Вывод результатов
    print(f"Accuracy: {accuracy}")
    if auc is not None:
        print(f"AUC: {auc}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Вывод информации о классах
    print(f"\nРаспределение классов в y_test:")
    print(y_test.value_counts())
