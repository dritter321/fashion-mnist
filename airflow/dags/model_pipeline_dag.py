from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2021, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG('MNIST Fashion pipeline',
          default_args=default_args,
          schedule_interval='@daily',
          catchup=False)

# Task 1
def preprocessing():
    import subprocess
    subprocess.run(["pip", "install", "-r", "/requirements.txt"])
    subprocess.run(["python", "/preprocessing/preprocessing.py"])

preprocess_task = PythonOperator(task_id='preprocessing',
                                 python_callable=preprocessing,
                                 dag=dag)

# Task 2
def train():
    import subprocess
    subprocess.run(["pip", "install", "-r", "/requirements.txt"])
    subprocess.run(["python", "/training/train.py"])

train_task = PythonOperator(task_id='training',
                            python_callable=train,
                            dag=dag)

# Task 3
def evaluate():
    import subprocess
    subprocess.run(["pip", "install", "-r", "/requirements.txt"])
    subprocess.run(["python", "/evaluation/eval.py"])

evaluate_task = PythonOperator(task_id='evaluation',
                               python_callable=evaluate,
                               dag=dag)

# Task 4
build_docker_image = BashOperator(
    task_id='build_docker_image',
    bash_command='docker build --build-arg EXPERIMENT_ID=737694674622074143 --build-arg RUN_ID=c09a656b6f624a72b6897ad6dcb7c122 -t my-flask-app .',
    dag=dag)

preprocess_task >> train_task >> evaluate_task >> build_docker_image