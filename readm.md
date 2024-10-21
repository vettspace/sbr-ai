
Структура проекта
Проект состоит из следующих файлов:
1.	data_preprocessing.py – скрипт для предобработки данных.
2.	feature_engineering.py – скрипт для создания и отбора признаков.
3.	model_training.py – скрипт для обучения модели машинного обучения.
4.	model_evaluation.py – скрипт для оценки и тестирования модели.
5.	model_inference.py – скрипт для интеграции модели и предсказания оттока клиентов на новых данных.
6.	data – папка, содержащая обучающие и тестовые данные (например, в формате CSV).


Описание задачи и инструкции
Задача: Построить и обучить модель машинного обучения для прогнозирования оттока клиентов в банке.
Цель: Проверить навыки Data Science и Data Engineering в предобработке данных, создании и обучении модели, а также в её интеграции и развертывании.
Задачи участника:
1.	Предобработка данных:
○	Осуществите очистку данных, включая удаление пропусков и преобразование категориальных признаков.
○	Разделите данные на обучающую и тестовую выборки.
2.	Feature Engineering:
○	Создайте новые признаки на основе имеющихся данных.
○	Произведите масштабирование признаков для улучшения качества модели.
3.	Обучение модели:
○	Обучите модель машинного обучения (например, RandomForest) на подготовленных данных.
○	Проведите гиперпараметрическую настройку модели.
4.	Оценка модели:
○	Оцените модель с использованием метрик точности и отчёта по классификации.
○	Проверьте модель на тестовых данных и проанализируйте результаты.
5.	Интеграция модели:
○	Реализуйте предсказание оттока для новых данных с использованием обученной модели.
○	Интегрируйте модель в систему, обеспечив возможность её использования на новых данных.
Критерии оценки:
●	Предобработка данных: Качество очистки и подготовки данных для обучения модели.
●	Feature Engineering: Эффективность создания и отбора признаков.
●	Обучение модели: Качество и точность обученной модели, а также её производительность.
●	Оценка модели: Полнота и корректность оценки модели на тестовых данных.
●	Интеграция модели: Способность модели эффективно работать с новыми данными и возможность её дальнейшего использования.

1. Общее описание задачи
Название задачи: прогнозирование оттока клиентов на основе исторических данных.
Цель задачи: проверить навыки участников в области анализа данных, разработки и оценки моделей машинного обучения, а также в обработке и визуализации данных.
Описание задачи: участникам предлагается построить модель машинного обучения для прогнозирования оттока клиентов на основе предоставленного набора исторических данных. Задача включает в себя этапы предобработки данных, выбора и тренировки модели, а также оценки ее точности. Дополнительно необходимо визуализировать результаты и объяснить значимость ключевых факторов, влияющих на отток.
2. Требования к исходным данным
Исходные данные:
●	Набор данных представляет собой таблицу с исторической информацией о клиентах банка.
●	Пример структуры данных:
○	CustomerID: Уникальный идентификатор клиента.
○	Age: Возраст клиента.
○	Gender: Пол клиента.
○	Tenure: Срок обслуживания клиента в банке (в годах).
○	Balance: Баланс на счету клиента.
○	NumOfProducts: Количество продуктов банка, используемых клиентом.
○	HasCrCard: Наличие кредитной карты (1 – да, 0 – нет).
○	IsActiveMember: Является ли клиент активным пользователем услуг банка (1 – да, 0 – нет).
○	EstimatedSalary: Оценка заработной платы клиента.
○	Exited: Целевой столбец, который указывает на то, покинул ли клиент банк (1 – да, 0 – нет).
Дополнительные данные:
●	в наборе данных могут присутствовать дополнительные признаки (features), такие как географическое положение клиента, информация о прошлых транзакциях, кредитный рейтинг и т.д.
●	Данные должны содержать пропуски, аномалии или некорректные значения, которые участники должны корректно обработать.
3. Задачи участника
1.	Предобработка данных:
○	Очистить данные от пропусков и аномальных значений.
○	Провести анализ данных для выявления корреляций между признаками.
○	Преобразовать категориальные переменные в числовые, если это необходимо.
○	Разделить данные на тренировочную и тестовую выборки.
2.	Построение модели:
○	Выбрать подходящую модель машинного обучения для задачи прогнозирования оттока (например, логистическая регрессия, случайный лес, градиентный бустинг и т.д.).
○	Обучить модель на тренировочной выборке.
○	Провести гиперпараметрическую настройку модели для повышения её точности.
○	Оценить производительность модели на тестовой выборке.
3.	Оценка и интерпретация модели:
○	Оценить модель с использованием различных метрик, таких как Accuracy, Precision, Recall, F1-score и ROC-AUC.
○	Визуализировать важность признаков (feature importance), влияющих на решение модели.
○	Подготовить краткий отчет с объяснением ключевых факторов, которые наиболее сильно влияют на отток клиентов.
4.	Визуализация данных:
○	Построить графики, иллюстрирующие распределение данных, корреляции между признаками, а также производительность модели.
○	Визуализировать кривую ROC и другие важные метрики для наглядной оценки модели.
4. Технические требования
Язык программирования и инструменты:
●	Язык программирования: Python.
●	Библиотеки для анализа данных: Pandas, NumPy.
●	Библиотеки для машинного обучения: Scikit-learn, XGBoost, LightGBM (или другие на выбор участника).
●	Библиотеки для визуализации: Matplotlib, Seaborn, Plotly (или другие на выбор участника).
●	Среда разработки: Jupyter Notebook или любая другая IDE на выбор участника.
Требования к коду:
●	Код должен быть хорошо структурированным, с комментариями и разделением на логические блоки.
●	Следует использовать лучшие практики в области анализа данных и машинного обучения.
●	Обязательно обеспечить воспроизводимость результатов (фиксация seed для случайных процессов, детальная инструкция по запуску кода).
5. Инструкция по выполнению
1.	Подготовка данных:
○	Участник должен загрузить предоставленный набор данных и провести его анализ.
○	Очистить и предобработать данные, подготовив их для обучения модели.
2.	Моделирование:
○	Выбрать и обучить модель на тренировочных данных.
○	Настроить гиперпараметры и оценить модель на тестовой выборке.
○	Провести интерпретацию модели и выделить важные признаки.
3.	Визуализация и отчет:
○	Построить графики, иллюстрирующие ключевые моменты анализа и результаты моделирования.
○	Подготовить краткий отчет, который включает описание процесса, результаты и выводы.
4.	Проверка и сдача работы:
○	Проверить, что все этапы выполнены корректно и результаты удовлетворяют требованиям задачи.
○	Подготовить проект к сдаче, оформив все необходимые документы и инструкции по запуску кода.