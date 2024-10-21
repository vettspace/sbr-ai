"""Интеграция модели и предсказание оттока клиентов на новых данных"""

import logging
import os

import joblib
import pandas as pd

from data_preprocessing import preprocess_data
from feature_engineering import feature_engineering

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {str(e)}")
        return None


def predict_churn(model, new_data):
    try:
        # Предобработка новых данных
        new_data = preprocess_data(new_data, is_training=False)
        new_data = feature_engineering(new_data, is_training=False)

        # Загрузка имен признаков и обеспечение соответствия
        feature_names = joblib.load('models/feature_names.pkl')
        new_data = new_data[feature_names]

        predictions = model.predict(new_data)
        probabilities = model.predict_proba(new_data)[:, 1]
        return predictions, probabilities
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {str(e)}")
        return None, None


if __name__ == "__main__":
    model = load_model('models/churn_model.pkl')
    if model is None:
        logger.error("Не удалось загрузить модель")
    else:
        try:
            new_data = pd.read_csv('data/new_customer_data.csv')

            predictions, probabilities = predict_churn(model, new_data)
            if predictions is not None and probabilities is not None:
                new_data['ChurnPrediction'] = predictions
                new_data['ChurnProbability'] = probabilities
                new_data.to_csv(
                    'data/new_customer_predictions.csv', index=False
                )
                logger.info("Предсказания успешно сохранены")

                # Вывод результатов
                for i, (pred, prob) in enumerate(
                    zip(predictions, probabilities), 1
                ):
                    logger.info(
                        f"Клиент {i}: Прогноз оттока = {pred}, Вероятность оттока = {prob:.2f}"
                    )
            else:
                logger.error("Не удалось выполнить предсказания")
        except FileNotFoundError:
            logger.error(
                "Файл 'new_customer_data.csv' не найден в директории 'data'"
            )
        except Exception as e:
            logger.error(f"Произошла ошибка при обработке данных: {str(e)}")
