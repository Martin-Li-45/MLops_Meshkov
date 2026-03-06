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
    df = pd.read_csv("/home/meshkov/airflow/dags/phones.csv")
    
    # === Очистка выбросов ===
    df = df[(df["Battery capacity (mAh)"] >= 500) & (df["Battery capacity (mAh)"] <= 10000)]
    df = df[(df["Screen size (inches)"] >= 3.0) & (df["Screen size (inches)"] <= 7.5)]
    df = df[(df["Resolution x"] >= 240) & (df["Resolution x"] <= 4000)]
    df = df[(df["Resolution y"] >= 320) & (df["Resolution y"] <= 4000)]
    df = df[(df["RAM (MB)"] >= 256) & (df["RAM (MB)"] <= 16384)]
    df = df[(df["Internal storage (GB)"] >= 1) & (df["Internal storage (GB)"] <= 1024)]
    df = df[(df["Rear camera"] >= 0) & (df["Rear camera"] <= 200)]
    df = df[(df["Front camera"] >= 0) & (df["Front camera"] <= 100)]
    df = df[df["Number of SIMs"] <= 3]
    df = df[(df["Price"] >= 1000) & (df["Price"] <= 500000)]
    
    # Бинарные признаки
    binary_columns = ['Touchscreen', 'Wi-Fi', 'Bluetooth', 'GPS', '3G', '4G/ LTE']
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "yes": 1, "no": 0})
    
    df = df.reset_index(drop=True)
    
    # Сохраняем очищенные данные БЕЗ плохого кодирования Name/Model
    df.to_csv('/home/meshkov/airflow/dags/df_clear.csv', index=False)
    print(f"Очистка завершена. Размер датасета: {df.shape}")
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

