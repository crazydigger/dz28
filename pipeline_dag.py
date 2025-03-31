from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime

# Определяем DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 10, 1),
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='A simple ML training pipeline',
    schedule_interval='@daily',
)

# Задача для запуска скрипта обучения модели
train_model_task = BashOperator(
    task_id='train_model',
    bash_command='python train.py',
    dag=dag,
)

# Определяем порядок выполнения задач (в данном случае только одна задача)
train_model_task
