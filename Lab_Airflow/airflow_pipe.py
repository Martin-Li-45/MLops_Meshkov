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
    df = pd.read_csv("/home/meshkov/airflow/dags/phones.csv", index_col=0)
    
    cat_columns = ['Name', 'Brand', 'Model', 'Processor', 'Operating system']
    num_columns = ['Battery capacity (mAh)', 'Screen size (inches)', 'Resolution x', 'Resolution y', 
                   'RAM (MB)', 'Internal storage (GB)', 'Rear camera', 'Front camera', 'Number of SIMs', 'Price']
    binary_columns = ['Touchscreen', 'Wi-Fi', 'Bluetooth', 'GPS', '3G', '4G/ LTE']
    
    # очистка
    df = df[df["Battery capacity (mAh)"].between(500, 10000)]
    df = df[df["Screen size (inches)"].between(3.0, 7.5)]
    df = df[df["Resolution x"].between(240, 4000)]
    df = df[df["Resolution y"].between(320, 4000)]
    df = df[df["RAM (MB)"].between(256, 16384)]
    df = df[df["Internal storage (GB)"].between(1, 1024)]
    df = df[df["Rear camera"].between(0, 200)]
    df = df[df["Front camera"].between(0, 100)]
    df = df[df["Number of SIMs"] <= 3]
    df = df[df["Price"].between(1000, 500000)]
    
    df = df.reset_index(drop=True)
    
    for col in binary_columns:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns])
    df[cat_columns] = ordinal.transform(df[cat_columns])
    
    # новые признаки
    df['PPI']                  = np.sqrt(df['Resolution x']**2 + df['Resolution y']**2) / df['Screen size (inches)']
    df['Total_Camera_MP']      = df['Rear camera'] + df['Front camera']
    df['Storage_per_RAM']      = df['Internal storage (GB)'] / (df['RAM (MB)'] / 1024)
    df['Battery_per_inch']     = df['Battery capacity (mAh)'] / df['Screen size (inches)']
    df['Screen_MP']            = (df['Resolution x'] * df['Resolution y']) / 1_000_000
    df['Multimedia_score']     = df['Screen_MP'] + df['Total_Camera_MP']
    
    df.to_csv('/home/meshkov/airflow/dags/df_clear.csv', index=False)
    print("Очищенный датасет сохранён →", df.shape)
    
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

