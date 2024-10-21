"""Скрипт для предобработки данных."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path):
    return pd.read_csv(
        file_path, sep=',', skipinitialspace=True, encoding='utf-8'
    )


def handle_outliers(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for column in numeric_columns:
        if column != 'Churn':  # Не обрабатываем столбец 'Churn'
            lower_bound = df[column].quantile(0.01)
            upper_bound = df[column].quantile(0.99)
            df[column] = np.clip(df[column], lower_bound, upper_bound)
    return df


def check_multicollinearity(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = [
        col for col in numeric_columns if col != 'Churn'
    ]  # Исключаем 'Churn'
    corr_matrix = df[numeric_columns].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    return df.drop(to_drop, axis=1)


def preprocess_data(df, is_training=True):
    # Если это обучающие данные, удаляем пропуски в целевой переменной
    if is_training and 'Churn' in df.columns:
        df = df.dropna(subset=['Churn'])

    # Обработка выбросов
    df = handle_outliers(df)

    # Преобразование категориальных признаков
    categorical_columns = df.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()
    if 'Churn' in categorical_columns:
        categorical_columns.remove('Churn')
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Проверка на мультиколлинеарность
    df = check_multicollinearity(df)

    return df


def split_data(df, target_column):
    if target_column not in df.columns:
        raise ValueError(
            f"Column '{target_column}' not found in the DataFrame. Available columns: {df.columns}"
        )
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)


if __name__ == "__main__":
    data = load_data('data/customer_data.csv')

    # Преобразование столбца 'Churn' в числовой тип
    if 'Churn' in data.columns:
        data['Churn'] = data['Churn'].astype(int)

    data = preprocess_data(data)

    if 'Churn' not in data.columns:
        raise ValueError("Column 'Churn' not found in the preprocessed data")

    X_train, X_test, y_train, y_test = split_data(data, 'Churn')

    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False, header=['Churn'])
    y_test.to_csv('data/y_test.csv', index=False, header=['Churn'])
