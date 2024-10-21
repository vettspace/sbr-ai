"""Скрипт для обучения модели машинного обучения."""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def train_model(X_train, y_train):
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(
            random_state=42, max_iter=1000
        ),
    }

    best_model = None
    best_score = 0

    # Проверим распределение классов в y_train
    class_counts = y_train.value_counts()
    min_class_samples = class_counts.min()

    if min_class_samples < 2:
        print(
            "Недостаточно образцов в одном из классов для выполнения кросс-валидации."
        )
        print("Обучение модели без кросс-валидации.")
        cv = None  # Не используем кросс-валидацию
    else:
        cv = min(5, min_class_samples)

    # Проверим, что y_train содержит не менее двух классов
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "Целевая переменная y_train должна содержать как минимум два класса."
        )

    y_train = (
        y_train.values.ravel()
    )  # Преобразуем в одномерный массив, если необходимо

    for name, model in models.items():
        try:
            if cv is not None:
                skf = StratifiedKFold(
                    n_splits=cv, shuffle=True, random_state=42
                )
                scores = []
                for train_index, test_index in skf.split(X_train, y_train):
                    X_fold_train, X_fold_test = (
                        X_train.iloc[train_index],
                        X_train.iloc[test_index],
                    )
                    y_fold_train, y_fold_test = (
                        y_train[train_index],
                        y_train[test_index],
                    )
                    model.fit(X_fold_train, y_fold_train)
                    score = model.score(X_fold_test, y_fold_test)
                    scores.append(score)
                avg_score = np.mean(scores)
                print(f"{name} average cross-validation score: {avg_score}")
            else:
                model.fit(X_train, y_train)
                avg_score = model.score(X_train, y_train)
                print(f"{name} training score: {avg_score}")

            if avg_score > best_score:
                best_score = avg_score
                best_model = model
        except Exception as e:
            print(f"Ошибка при оценке модели {name}: {e}")

    if best_model is None:
        print("Не удалось найти подходящую модель.")
        return None

    # Настройка гиперпараметров для лучшей модели
    if isinstance(best_model, RandomForestClassifier):
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
        }
    elif isinstance(best_model, GradientBoostingClassifier):
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
        }
    else:  # LogisticRegression
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs'],
        }

    if cv is not None:
        grid_search = GridSearchCV(
            best_model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model.fit(X_train, y_train)

    return best_model


if __name__ == "__main__":
    X_train = pd.read_csv('data/X_train_fe.csv')
    y_train = pd.read_csv('data/y_train.csv')['Churn']

    # Проверяем, что число образцов в X_train и y_train совпадает
    assert len(X_train) == len(
        y_train
    ), "Число образцов в X_train и y_train не совпадает."

    # Проверяем, что y_train содержит не менее двух классов
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "Целевая переменная y_train должна содержать как минимум два класса."
        )

    model = train_model(X_train, y_train)
    if model is not None:
        # Создаем директорию models, если она не существует
        if not os.path.exists('models'):
            os.makedirs('models')
        joblib.dump(model, 'models/churn_model.pkl')

        # Сохраняем имена признаков
        feature_names = X_train.columns.tolist()
        joblib.dump(feature_names, 'models/feature_names.pkl')