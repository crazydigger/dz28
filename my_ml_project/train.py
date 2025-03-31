import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Загружаем данные
def load_data():
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                       header=None,
                       names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    return data


# Основная функция для обучения модели
def train_model():
    # Загружаем данные
    data = load_data()

    # Делим данные на обучающую и тестовую выборки
    X = data.iloc[:, :-1]  # Все столбцы, кроме последнего
    y = data.iloc[:, -1]  # Последний столбец (целевой)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Определяем гиперпараметры модели
    n_estimators = 100
    max_depth = 3

    # Логируем параметры в MLFlow
    mlflow.start_run()
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Обучаем модель
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)

    # Делаем предсказания и вычисляем метрики
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Логируем метрики в MLFlow
    mlflow.log_metric("accuracy", accuracy)

    # Сохраняем модель в MLFlow
    mlflow.sklearn.log_model(model, "model")

    print(f"Model trained with accuracy: {accuracy}")


if __name__ == "__main__":
    train_model()