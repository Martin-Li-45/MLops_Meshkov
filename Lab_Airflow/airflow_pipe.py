# airflow_phones.py
import pandas as pd
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta
from train_phones import train

def download_data():
    """Загружает датасет с мобильными телефонами."""
    # Используем локальный файл, так как в предоставленных данных был CSV.
    # В реальности URL может быть другим.
    df = pd.read_csv('ndtv_data_final.csv', index_col=0)
    df.to_csv("phones.csv", index=False)
    print("df shape: ", df.shape)
    return df

def clear_data():
    """Выполняет очистку данных для датасета с телефонами."""
    df = pd.read_csv("phones.csv")

    # Удаляем строки с отсутствующими значениями в ключевых колонках
    initial_shape = df.shape[0]
    df = df.dropna(subset=['Price', 'RAM (MB)', 'Internal storage (GB)'])
    print(f"Удалено строк с пропусками: {initial_shape - df.shape[0]}")

    # Преобразуем цену в числовой формат, удаляем возможные символы валют
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df.dropna(subset=['Price'])

    # Фильтрация на основе здравого смысла и анализа данных
    # Цена не может быть меньше 500 (условный порог для бюджетного телефона)
    question_price_low = df[df["Price"] < 500]
    df = df.drop(question_price_low.index)
    print(f"Удалено строк с ценой < 500: {question_price_low.shape[0]}")

    # Цена не может быть больше 200000 (условный порог для флагманов)
    question_price_high = df[df["Price"] > 200000]
    df = df.drop(question_price_high.index)
    print(f"Удалено строк с ценой > 200000: {question_price_high.shape[0]}")

    # Оперативная память (RAM) не может быть меньше 128 МБ
    question_ram_low = df[df["RAM (MB)"] < 128]
    df = df.drop(question_ram_low.index)
    print(f"Удалено строк с RAM < 128MB: {question_ram_low.shape[0]}")

    # Внутренняя память не может быть меньше 1 ГБ
    question_storage_low = df[df["Internal storage (GB)"] < 1]
    df = df.drop(question_storage_low.index)
    print(f"Удалено строк с Storage < 1GB: {question_storage_low.shape[0]}")

    # Сброс индекса после всех удалений
    df = df.reset_index(drop=True)

    # Сохраняем очищенный датасет. Категориальные поля остаются как есть для дальнейшей обработки.
    df.to_csv('phones_clear.csv', index=False)
    print(f"Итоговый размер датасета: {df.shape}")
    return True

# Определение DAG
dag_phones = DAG(
    dag_id="phone_price_prediction",
    start_date=datetime(2025, 2, 3),
    max_active_tasks=4,
    schedule=timedelta(minutes=5),
    max_active_runs=1,
    catchup=False,
)

# Определение задач
download_task = PythonOperator(
    python_callable=download_data,
    task_id="download_phones",
    dag=dag_phones
)

clear_task = PythonOperator(
    python_callable=clear_data,
    task_id="clear_phones",
    dag=dag_phones
)

train_task = PythonOperator(
    python_callable=train,
    task_id="train_phones",
    dag=dag_phones
)

# Установка последовательности задач
download_task >> clear_task >> train_task