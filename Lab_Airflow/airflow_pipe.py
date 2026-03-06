import pandas as pd
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from train_model import train

def download_data():
    df = pd.read_csv('/home/meshkov/airflow/dags/ndtv_data_final.csv', delimiter=',')
    df.to_csv("/home/meshkov/airflow/dags/phones.csv", index=False)
    print("Данные загружены, размер:", df.shape)
    return df

def clear_data():
    # Загружаем данные
    df = pd.read_csv("/home/meshkov/airflow/dags/phones.csv", index_col=0)
    
    # Определяем типы колонок
    cat_columns = ['Brand', 'Processor', 'Operating system']          # Name и Model убираем из кодирования
    num_columns = ['Battery capacity (mAh)', 'Screen size (inches)', 'Resolution x', 'Resolution y', 
                   'RAM (MB)', 'Internal storage (GB)', 'Rear camera', 'Front camera', 
                   'Number of SIMs', 'Price']
    binary_columns = ['Touchscreen', 'Wi-Fi', 'Bluetooth', 'GPS', '3G', '4G/ LTE']
    
    # Сначала преобразуем Yes/No → 1/0 (до удаления строк!)
    for col in binary_columns:
        df[col] = df[col].map({"Yes": 1, "No": 0, "yes": 1, "no": 0}).fillna(0).astype(int)
    
    # ────────────────────────────────────────────────────────────────
    # Фильтрация выбросов (оставляем почти как было)
    # ────────────────────────────────────────────────────────────────
    
    # Батарея
    df = df[df["Battery capacity (mAh)"].between(500, 10000)]
    
    # Экран
    df = df[df["Screen size (inches)"].between(3.0, 7.5)]
    
    # Разрешение
    df = df[df["Resolution x"].between(240, 4000)]
    df = df[df["Resolution y"].between(320, 4000)]
    
    # Оперативка
    df = df[df["RAM (MB)"].between(256, 16384)]
    
    # Встроенная память
    df = df[df["Internal storage (GB)"].between(1, 1024)]
    
    # Камеры
    df = df[df["Rear camera"].between(0, 200)]
    df = df[df["Front camera"].between(0, 100)]
    
    # SIM-карты
    df = df[df["Number of SIMs"] <= 3]
    
    # Цена
    df = df[df["Price"].between(1000, 500000)]
    
    # Сбрасываем индекс после всех удалений
    df = df.reset_index(drop=True)
    
    # НЕ кодируем Name, Model, Brand, Processor, OS здесь!
    # Оставляем их строками → закодирует ColumnTransformer / OneHotEncoder / CatBoost
    
    # Сохраняем очищенные "сырые" данные
    df.to_csv('/home/meshkov/airflow/dags/df_clear.csv', index=False)
    print("Очищенный датасет сохранён в df_clear.csv")
    print(f"Размер после очистки: {df.shape}")
    
    return True
    
dag_phones = DAG(
    dag_id="train_pipe",
    start_date=datetime(2025, 2, 3),
    schedule=timedelta(minutes=5),
    max_active_runs=1,
    catchup=False,
)

download_task = PythonOperator(python_callable=download_data, task_id="download_phones", dag=dag_phones)
clear_task = PythonOperator(python_callable=clear_data, task_id="clear_phones", dag=dag_phones)
train_task = PythonOperator(python_callable=train, task_id="train_phones", dag=dag_phones)

download_task >> clear_task >> train_task

