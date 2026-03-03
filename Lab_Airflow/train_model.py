# train_phones.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from mlflow.models import infer_signature
import joblib

def eval_metrics(actual, pred):
    """Вычисляет метрики качества регрессии."""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def prepare_features(frame):
    """
    Подготавливает признаки для модели: масштабирует числа и кодирует категории.
    Возвращает обработанные X, y и объект PowerTransformer для обратного преобразования цены.
    """
    df = frame.copy()
    target = 'Price'

    # Определение колонок
    numerical_cols = ['RAM (MB)', 'Internal storage (GB)', 'Battery capacity (mAh)',
                      'Screen size (inches)', 'Rear camera', 'Front camera']
    # Выбираем только те колонки, которые есть в датафрейме
    numerical_cols = [col for col in numerical_cols if col in df.columns]

    categorical_cols = ['Brand', 'Operating system', 'Touchscreen', 'Wi-Fi', 'Bluetooth', 'GPS']
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    # Разделение на X и y
    X = df[numerical_cols + categorical_cols]
    y = df[target]

    # Предобработка числовых колонок: заполнение пропусков средним, затем масштабирование
    # (В реальном проекте можно использовать SimpleImputer в пайплайне)
    for col in numerical_cols:
        if X[col].isnull().any():
            X[col].fillna(X[col].mean(), inplace=True)

    # Преобразование целевой переменной
    power_trans = PowerTransformer(method='yeo-johnson')
    y_transformed = power_trans.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Создание препроцессора
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # Применяем препроцессор к X
    X_processed = preprocessor.fit_transform(X)

    # Сохраняем препроцессор для дальнейшего использования (например, для инференса)
    # В этом примере мы его не возвращаем, но в реальном проекте это было бы полезно.
    # Для простоты сейчас мы просто возвращаем обработанные данные.
    # joblib.dump(preprocessor, 'preprocessor.pkl')

    return X_processed, y_transformed, power_trans

def train():
    """Загружает данные, обучает модель и логирует результаты в MLflow."""
    df = pd.read_csv("./phones_clear.csv")

    # Подготовка признаков
    X, y_transformed, power_trans = prepare_features(df)

    # Разделение на train и validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_transformed, test_size=0.3, random_state=42
    )

    # Параметры для GridSearchCV
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.01, 0.2],
        "penalty": ["l1", "l2", "elasticnet"],
        "loss": ['squared_error', 'huber', 'epsilon_insensitive'],
        "fit_intercept": [False, True],
    }

    mlflow.set_experiment("Phone Price Prediction Model")
    with mlflow.start_run():
        lr = SGDRegressor(random_state=42, max_iter=1000, tol=1e-3)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=4, scoring='neg_mean_squared_error')
        clf.fit(X_train, y_train)

        best = clf.best_estimator_
        y_pred_transformed = best.predict(X_val)

        # Обратное преобразование цены для вычисления метрик в исходном масштабе
        y_val_original = power_trans.inverse_transform(y_val.reshape(-1, 1))
        y_pred_original = power_trans.inverse_transform(y_pred_transformed.reshape(-1, 1))

        (rmse, mae, r2) = eval_metrics(y_val_original, y_pred_original)

        # Логирование параметров и метрик
        mlflow.log_param("alpha", best.alpha)
        mlflow.log_param("l1_ratio", best.l1_ratio)
        mlflow.log_param("penalty", best.penalty)
        mlflow.log_param("loss", best.loss)
        mlflow.log_param("fit_intercept", best.fit_intercept)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Логирование модели
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)

        # Сохранение модели локально (опционально)
        with open("sgd_phones.pkl", "wb") as file:
            joblib.dump(best, file)

        print(f"Model trained. RMSE: {rmse:.2f}, R2: {r2:.4f}, MAE: {mae:.2f}")

if __name__ == "__main__":
    train()