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

def clear_data_phones():
    # Загружаем данные
    df = pd.read_csv("ndtv_data_final.csv", index_col=0)
    
    # Определяем типы колонок
    cat_columns = ['Brand', 'Model', 'Processor', 'Operating system']
    num_columns = ['Battery capacity (mAh)', 'Screen size (inches)', 'Resolution x', 'Resolution y', 
                   'RAM (MB)', 'Internal storage (GB)', 'Rear camera', 'Front camera', 'Number of SIMs', 'Price']
    binary_columns = ['Touchscreen', 'Wi-Fi', 'Bluetooth', 'GPS', '3G', '4G/ LTE']
    
    # Анализ и очистка данных
    
    # Удаляем слишком маленькую батарею (менее 500 mAh)
    question_battery = df[df["Battery capacity (mAh)"] < 500]
    df = df.drop(question_battery.index)
    
    # Удаляем слишком большую батарею (более 10000 mAh)
    question_battery = df[df["Battery capacity (mAh)"] > 10000]
    df = df.drop(question_battery.index)
    
    # Удаляем слишком маленький экран (менее 3 дюймов)
    question_screen = df[df["Screen size (inches)"] < 3.0]
    df = df.drop(question_screen.index)
    
    # Удаляем слишком большой экран (более 7.5 дюймов)
    question_screen = df[df["Screen size (inches)"] > 7.5]
    df = df.drop(question_screen.index)
    
    # Удаляем слишком маленькое разрешение по ширине
    question_res = df[df["Resolution x"] < 240]
    df = df.drop(question_res.index)
    
    # Удаляем слишком большое разрешение по ширине
    question_res = df[df["Resolution x"] > 4000]
    df = df.drop(question_res.index)
    
    # Удаляем слишком маленькое разрешение по высоте
    question_res = df[df["Resolution y"] < 320]
    df = df.drop(question_res.index)
    
    # Удаляем слишком большое разрешение по высоте
    question_res = df[df["Resolution y"] > 4000]
    df = df.drop(question_res.index)
    
    # Удаляем слишком маленькую RAM (менее 256 MB)
    question_ram = df[df["RAM (MB)"] < 256]
    df = df.drop(question_ram.index)
    
    # Удаляем слишком большую RAM (более 16 GB = 16384 MB)
    question_ram = df[df["RAM (MB)"] > 16384]
    df = df.drop(question_ram.index)
    
    # Удаляем слишком маленькую память (менее 1 GB)
    question_storage = df[df["Internal storage (GB)"] < 1]
    df = df.drop(question_storage.index)
    
    # Удаляем слишком большую память (более 1 TB = 1024 GB)
    question_storage = df[df["Internal storage (GB)"] > 1024]
    df = df.drop(question_storage.index)
    
    # Удаляем отрицательные значения основной камеры
    question_camera = df[df["Rear camera"] < 0]
    df = df.drop(question_camera.index)
    
    # Удаляем слишком большую основную камеру (более 200 MP)
    question_camera = df[df["Rear camera"] > 200]
    df = df.drop(question_camera.index)
    
    # Удаляем отрицательные значения фронтальной камеры
    question_camera = df[df["Front camera"] < 0]
    df = df.drop(question_camera.index)
    
    # Удаляем слишком большую фронтальную камеру (более 100 MP)
    question_camera = df[df["Front camera"] > 100]
    df = df.drop(question_camera.index)
    
    # Удаляем слишком много SIM-карт (более 3)
    question_sim = df[df["Number of SIMs"] > 3]
    df = df.drop(question_sim.index)
    
    # Удаляем слишком дешёвые телефоны (менее 1000 рублей)
    question_price = df[df["Price"] < 1000]
    df = df.drop(question_price.index)
    
    # Удаляем слишком дорогие телефоны (более 500000 рублей)
    question_price = df[df["Price"] > 500000]
    df = df.drop(question_price.index)
    
    # Сбрасываем индекс после удаления строк
    df = df.reset_index(drop=True)
    
    # Кодируем бинарные признаки (Yes/No -> 1/0)
    for col in binary_columns:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    
    # Кодируем категориальные признаки
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns])
    ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    
    # Сохраняем очищенный датасет
    df.to_csv('df_phones_clear.csv')
    print("Очищенный датасет сохранён в df_phones_clear.csv")
    print(f"Размер после очистки: {df.shape}")
    
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