import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer
import numpy as np
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from pathlib import Path
from datetime import timedelta
from train_model import train

def download_data():
    # Используем локальный файл или загружаем из URL
    df = pd.read_csv('ndtv_data_final.csv')
    df.to_csv("smartphones.csv", index=False)
    print(f"df: {df.shape}")
    return df

def clear_data():
    df = pd.read_csv("smartphones.csv")
    
    # Категориальные и числовые колонки для смартфонов
    cat_columns = ['Brand', 'Model', 'Operating system', 'Touchscreen', 
                   'Wi-Fi', 'Bluetooth', 'GPS', '3G', '4G/ LTE']
    num_columns = ['Battery capacity (mAh)', 'Screen size (inches)', 
                   'Resolution x', 'Resolution y', 'Processor', 
                   'RAM (MB)', 'Internal storage (GB)', 
                   'Rear camera', 'Front camera', 'Number of SIMs', 'Price']
    
    # Очистка данных - удаление выбросов
    
    # Слишком маленькая батарея
    question_battery = df[df['Battery capacity (mAh)'] < 1000]
    df = df.drop(question_battery.index)
    
    # Слишком большая батарея (нереалистично)
    question_battery = df[df['Battery capacity (mAh)'] > 10000]
    df = df.drop(question_battery.index)
    
    # Слишком маленький экран
    question_screen = df[df['Screen size (inches)'] < 3.0]
    df = df.drop(question_screen.index)
    
    # Слишком большой экран (нереалистично)
    question_screen = df[df['Screen size (inches)'] > 10.0]
    df = df.drop(question_screen.index)
    
    # Слишком низкая цена
    question_price = df[df['Price'] < 1000]
    df = df.drop(question_price.index)
    
    # Слишком высокая цена (выбросы)
    question_price = df[df['Price'] > 200000]
    df = df.drop(question_price.index)
    
    # Слишком мало RAM
    question_ram = df[df['RAM (MB)'] < 512]
    df = df.drop(question_ram.index)
    
    # Слишком много RAM (нереалистично для датасета)
    question_ram = df[df['RAM (MB)'] > 20000]
    df = df.drop(question_ram.index)
    
    df = df.reset_index(drop=True)
    
    # Кодирование категориальных переменных
    ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    ordinal.fit(df[cat_columns])
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    
    # Заполнение пропусков (если есть)
    df = df.fillna(df.median(numeric_only=True))
    
    df.to_csv('df_clear.csv', index=False)
    print(f"Cleaned data: {df.shape}")
    return True

# DAG конфигурация
dag_cars = DAG(
    dag_id="phone_price_prediction",
    start_date=datetime(2025, 2, 3),
    max_active_tasks=4,
    schedule=timedelta(minutes=30),
    max_active_runs=1,
    catchup=False,
)

download_task = PythonOperator(
    python_callable=download_data, 
    task_id="download_smartphones", 
    dag=dag_cars
)

clear_task = PythonOperator(
    python_callable=clear_data, 
    task_id="clear_smartphones", 
    dag=dag_cars
)

train_task = PythonOperator(
    python_callable=train, 
    task_id="train_smartphones", 
    dag=dag_cars
)

download_task >> clear_task >> train_task